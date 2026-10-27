#version 450 core
#extension GL_EXT_samplerless_texture_functions : require

// A multisample depth image's first sample, for single-sample texture reads of it
// (BlitHelper::CopyMsDepthSample0, KYTY_MS_DEPTH_SAMPLE0).
layout(binding = 0, set = 0) uniform texture2DMS depth;

void main() {
	gl_FragDepth = texelFetch(depth, ivec2(gl_FragCoord.xy), 0).r;
}
