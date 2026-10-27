// Exercise the AJM Opus decoder with output buffers smaller than a packet.
#include "libs/ajm/opus_decoder.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numbers>
#include <opus.h>
#include <vector>

using namespace Libs::Audio::Ajm;

namespace {

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "AjmOpusTests: %s\n", message);
		std::abort();
	}
}

constexpr uint32_t PacketSamples = 960;             // 20 ms at 48 kHz
constexpr size_t   SampleBytes   = sizeof(int16_t); // mono, 16-bit

// A 440 Hz tone as Opus packets, each preceded by the 2-byte little-endian size AJM expects.
std::vector<uint8_t> EncodeStream(uint32_t packets) {
	int   error   = 0;
	auto* encoder = opus_encoder_create(48000, 1, OPUS_APPLICATION_AUDIO, &error);
	Check(encoder != nullptr && error == OPUS_OK, "the Opus encoder could not be created");
	std::vector<uint8_t> stream;
	std::vector<int16_t> pcm(PacketSamples);
	for (uint32_t p = 0; p < packets; p++) {
		for (uint32_t i = 0; i < PacketSamples; i++) {
			const auto t = static_cast<double>(p * PacketSamples + i) / 48000.0;
			pcm[i] = static_cast<int16_t>(8000.0 * std::sin(2.0 * std::numbers::pi * 440.0 * t));
		}
		uint8_t    packet[1500];
		const auto size = opus_encode(encoder, pcm.data(), static_cast<int>(PacketSamples), packet,
		                              sizeof(packet));
		Check(size > 0, "a packet could not be encoded");
		stream.push_back(static_cast<uint8_t>(size & 0xff));
		stream.push_back(static_cast<uint8_t>(size >> 8));
		stream.insert(stream.end(), packet, packet + size);
	}
	opus_encoder_destroy(encoder);
	return stream;
}

std::unique_ptr<AjmOpusDecoder> MakeDecoder() {
	auto decoder = std::make_unique<AjmOpusDecoder>(1, AjmSampleEncoding::S16);
	const AjmDecOpusInitializeParameters params {1, 48000, 0};
	Check(decoder->Initialize(&params, sizeof(params)).result == OK,
	      "the decoder could not be initialized");
	return decoder;
}

std::vector<uint8_t> DecodeAtOnce(const std::vector<uint8_t>& stream, AjmGaplessState* gapless) {
	auto decoder = MakeDecoder();
	// More than enough: every packet takes more than 2 bytes of the stream.
	std::vector<uint8_t> output(stream.size() / 2 * PacketSamples * SampleBytes);
	const auto           result =
	    decoder->Decode(stream.data(), stream.size(), output.data(), output.size(), true, gapless);
	// Running out of input reports PARTIAL_INPUT even when every packet was decoded.
	Check((result.result == OK || result.result == AJM_RESULT_PARTIAL_INPUT) &&
	          result.input_consumed == stream.size(),
	      "the reference decode failed");
	output.resize(result.output_written);
	return output;
}

std::vector<uint8_t> DecodeInPieces(const std::vector<uint8_t>& stream, size_t room,
                                    bool multiple_frames, AjmGaplessState* gapless,
                                    size_t expected_bytes) {
	auto                 decoder = MakeDecoder();
	std::vector<uint8_t> decoded;
	std::vector<uint8_t> output(room);
	size_t               consumed = 0;
	while (decoded.size() < expected_bytes) {
		const auto result = decoder->Decode(stream.data() + consumed, stream.size() - consumed,
		                                    output.data(), output.size(), multiple_frames, gapless);
		Check(result.result == OK || result.result == AJM_RESULT_PARTIAL_INPUT,
		      "a call with room failed");
		Check(result.output_written != 0, "a call with room wrote nothing");
		Check(gapless == nullptr || !gapless->IsEnd() ||
		          decoded.size() + result.output_written == expected_bytes,
		      "the stream ended before its last samples were written");
		consumed += result.input_consumed;
		decoded.insert(decoded.end(), output.begin(),
		               output.begin() + static_cast<std::ptrdiff_t>(result.output_written));
	}
	Check(consumed == stream.size(), "not all of the input was consumed");
	return decoded;
}

void TestSmallOutputs() {
	const auto stream   = EncodeStream(8);
	const auto expected = DecodeAtOnce(stream, nullptr);
	Check(expected.size() == 8 * PacketSamples * SampleBytes, "the reference has the wrong length");
	for (const bool multiple_frames: {false, true}) {
		// Less than a packet, ending inside a sample, one packet, more than one packet.
		for (const size_t room: {size_t {1024}, size_t {1001}, size_t {1920}, size_t {3000}}) {
			Check(DecodeInPieces(stream, room, multiple_frames, nullptr, expected.size()) ==
			          expected,
			      "small outputs changed the decoded samples");
		}
	}
}

void TestGaplessSmallOutputs() {
	const auto      stream = EncodeStream(6);
	AjmGaplessState reference;
	reference.Set({5000, 312, 0}, true);
	const auto expected = DecodeAtOnce(stream, &reference);
	Check(expected.size() == 5000 * SampleBytes && reference.IsEnd(),
	      "the gapless reference has the wrong length");
	for (const bool multiple_frames: {false, true}) {
		AjmGaplessState gapless;
		gapless.Set({5000, 312, 0}, true);
		const auto decoded =
		    DecodeInPieces(stream, 1024, multiple_frames, &gapless, expected.size());
		Check(decoded == expected && gapless.IsEnd(),
		      "small outputs changed the skip or the sample limit");
	}
}

void TestNoRoom() {
	const auto stream  = EncodeStream(2);
	auto       decoder = MakeDecoder();
	uint8_t    output[1] {};
	for (const size_t room: {size_t {0}, size_t {1}}) {
		const auto result =
		    decoder->Decode(stream.data(), stream.size(), output, room, true, nullptr);
		Check(result.result == AJM_RESULT_NOT_ENOUGH_ROOM && result.input_consumed == 0 &&
		          result.output_written == 0,
		      "without room for a sample, a packet was consumed");
	}
	const auto result = decoder->Decode(stream.data(), stream.size(), nullptr, 1024, true, nullptr);
	Check(result.result == AJM_RESULT_NOT_ENOUGH_ROOM && result.input_consumed == 0,
	      "without an output, a packet was consumed");
}

void TestResetDropsLeftoverSamples() {
	const auto           stream   = EncodeStream(2);
	const auto           expected = DecodeAtOnce(stream, nullptr);
	auto                 decoder  = MakeDecoder();
	std::vector<uint8_t> output(1024);
	auto                 result =
	    decoder->Decode(stream.data(), stream.size(), output.data(), output.size(), false, nullptr);
	Check(result.result == OK && result.output_written == output.size(),
	      "the first piece is missing");
	decoder->Reset();
	result =
	    decoder->Decode(stream.data(), stream.size(), output.data(), output.size(), false, nullptr);
	Check(result.result == OK && std::memcmp(output.data(), expected.data(), output.size()) == 0,
	      "a reset kept samples of the previous packet");
}

} // namespace

int main() {
	TestSmallOutputs();
	TestGaplessSmallOutputs();
	TestNoRoom();
	TestResetDropsLeftoverSamples();
	std::puts("AjmOpusTests: all cases passed");
}
