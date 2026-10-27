#version 450 core
#extension GL_EXT_samplerless_texture_functions : require

// The first sample of a multisample depth/stencil image's stencil, into an R8_UINT image
// (BlitHelper::CopyMsStencilSample0, KYTY_MS_DEPTH_SAMPLE0).
layout(binding = 0, set = 0) uniform utexture2DMS stencil;
layout(location = 0) out uint result;

void main() {
	result = texelFetch(stencil, ivec2(gl_FragCoord.xy), 0).r;
}
