// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: MIT

#include <donut/shaders/forward_vertex.hlsli>

// Keep the production VS signatures, including unused fields, so semantic registers match.
float4 depth(float4 position : SV_Position, float2 texCoord : TEXCOORD) : SV_Target0
{
    return float4(texCoord, 0.0, 1.0);
}

float4 scene(float4 position : SV_Position, SceneVertex vertex) : SV_Target0
{
    return float4(vertex.texCoord, 0.0, 1.0);
}
