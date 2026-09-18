# Texture coordinate storage

Donut supports `TexCoordFormat::Float32` (the default), `TexCoordFormat::Float16`, and `TexCoordFormat::Unorm16` for GPU texture coordinates. `Float16` stores each component as an IEEE half float. `Unorm16` automatically normalizes source UV bounds, stores two unsigned normalized 16-bit components, and reconstructs the original UVs with generated FP32 scale and offset. Shader interpolation and material evaluation still use `float2`.

To use normalized UNORM16 for subsequent scene imports, set the default before loading:

```cpp
scene.SetDefaultTexCoordFormat(donut::engine::TexCoordFormat::Unorm16);
scene.Load("scene.json");
```

This also applies to `LoadWithThreadPool`. Changing the default does not convert existing buffer groups. To override an imported or manually created group, assign its format before its first GPU upload through `FinishedLoading`, `RefreshBuffers`, or `Refresh`:

```cpp
mesh->buffers->texCoordFormat = donut::engine::TexCoordFormat::Float32;
```

All meshes sharing a `BufferGroup` share this setting. Both UV streams use the same format. CPU-side `texcoord1Data` and `texcoord2Data` remain `std::vector<dm::float2>`; packing happens during upload. Do not change `texCoordFormat` after the vertex buffer has been created.

| GPU format | Bytes per UV pair | Input assembler format | Decode metadata |
| --- | ---: | --- | --- |
| `Float32` | 8 | `RG32_FLOAT` | None |
| `Float16` | 4 | `RG16_FLOAT` | None |
| `Unorm16` | 4 | `RG16_UNORM` | FP32 U/V scale and offset, 16 bytes per UV set |

Both 16-bit formats halve the vertex UV payload. Each attribute range remains aligned to 16 bytes, so total allocation savings depend on vertex count and the other attributes. Use `BufferGroup::getTexCoordStride()` for CPU offset calculations. Skinned buffers inherit the prototype's actual format and decode metadata, and skinning copies the stored UV bits unchanged.

For FP16, a nonfinite component or a component with absolute value greater than 65504 causes upload to log a warning and fall back to FP32 for the entire group. This range check does not guarantee sufficient precision: negative and tiled UVs are supported, but large UV magnitudes retain less fractional detail.

## Automatic UNORM16 bounds

The same source asset can use any format; no additional authored UV transforms or asset metadata are required. During upload, Donut computes each UV set's minimum and range, then packs each component as follows:

```text
scale = maximumUV - minimumUV
offset = minimumUV
packedUV = round(clamp((sourceUV - offset) / scale, 0, 1) * 65535)
decodedUV = (packedUV / 65535) * scale + offset
```

A zero-width axis uses scale zero and packed value zero, reconstructing its constant offset without division by zero. Negative, tiled, and large-offset UVs do not need to fit into `[0,1]` in the asset. For example, U bounds `[3000,3002]` produce scale `2` and offset `3000`; only the packed coordinates occupy `[0,1]`.

Bounds are computed independently per mesh vertex range and UV set. Disjoint meshes sharing one `BufferGroup` retain separate bounds. Meshes with overlapping vertex ranges share merged bounds so each stored vertex has a consistent decode. Unreferenced vertices receive separate ranges. `BufferGroup::texCoordDecodeRanges` retains the generated ranges, and `getTexCoordDecodeRange(vertexIndex)` looks them up using a buffer-relative vertex index. Skinned buffers rebase these ranges to their copied vertices.

The FP32 scale and offset are shared metadata, not part of each vertex. `GeometryData` contains a `float4` for each UV set (`scale.xy`, `offset.zw`), adding 32 bytes to each geometry record. The raster passes pass the first UV set's decode in draw constants. Thus actual memory savings include this metadata overhead as well as vertex-buffer alignment.

UNORM16 falls back to FP32 for the entire group if either stream contains nonfinite values or finite FP32 decode bounds cannot be represented safely. Inspect `texCoordFormat` after upload for the actual format. Outliers enlarge the bounds and reduce precision for other vertices in the same range. The ideal per-component quantization error is at most `scale / (2 * 65535)`, plus FP32 reconstruction rounding; large offsets still face ordinary FP32 precision limits. There is no automatic visual-quality threshold. Keep groups in FP32 when texture detail or alpha-tested edges require it.

## Renderer and application integration

Donut's depth, forward, and GBuffer passes support all three formats through input assembler and buffer-load paths. The material ID pass inherits the GBuffer buffer-load support. Custom subclasses that customize input layouts should override `CreateInputLayout(vertexShader, params, texCoordFormat)` to supply matching layouts for each format. The original two-argument overload remains available for existing FP32 subclasses. UNORM16 input assembler attributes already arrive as normalized floats; apply the generated scale and offset once with `DecodeTexCoord`.

Application-specific ray tracing shaders and custom vertex-buffer readers require a separate migration before enabling a 16-bit format. `GeometryData::texCoordFormat` records the actual encoding (`0` for FP32, `1` for FP16, `2` for UNORM16). Include `donut/shaders/bindless.h` and use its shared geometry loader after checking for absent attributes:

```hlsl
float2 LoadGeometryUV(ByteAddressBuffer vertexBuffer, GeometryData geometry, uint vertexIndex)
{
    if (geometry.texCoord1Offset == ~0u)
        return float2(0, 0);
    return LoadGeometryTexCoord(vertexBuffer, geometry, vertexIndex, 0);
}
```

Here `vertexBuffer` is the buffer selected by `geometry.vertexBufferIndex`, and `vertexIndex` is relative to the geometry: its byte offset already includes mesh and geometry offsets. Pass `1` as the final argument for the second UV set, guarding `texCoord2Offset` in the same way. Alternatively, call `LoadTexCoord(buffer, offset, vertexIndex, format, scaleBias)` after checking that the offset is not `~0u`. The older four-argument `LoadTexCoord` returns normalized `[0,1]` coordinates for UNORM16 and therefore is insufficient to reconstruct source UVs. Use `GetTexCoordStride(geometry.texCoordFormat)` for other shader offset calculations. The legacy `c_SizeOfTexcoord` name is preserved at 8 bytes per FP32 UV pair; `c_SizeOfTexcoord16` is 4 bytes per FP16 or UNORM16 UV pair.

The decode is affine, so it can be applied before interpolation or after barycentric interpolation when all vertices share a decode range. UV-dependent material operations must use decoded coordinates. When using explicit gradients computed from packed coordinates, multiply those gradients by the decode scale.

Rebuild Donut and application shader binaries together. `GeometryData` grows from 64 to 96 bytes, and depth, forward, and GBuffer push constants grow from 32 to 48 bytes. These ABI changes also affect applications that keep using FP32; stale compiled shaders and hard-coded structure strides are incompatible. The storage paths use ordinary 32-bit shader operations for decoding and do not require native 16-bit shader arithmetic.

## Tests

With `DONUT_WITH_UNIT_TESTS=ON`, build `donut_shaders` and `donut_all_tests`. `test_texcoords` runs CPU layout checks by default. Supply the compiled shader directories to run the GPU tests:

```text
test_texcoords -dx12 --gpu <Donut shaders>/dxil
test_texcoord_raster -dx12 --gpu <Donut shaders>/dxil --test-shaders <Donut test build>/shaders/dxil
```

Use `-dx11` and `dxbc` for DX11, or `-vk` and `spirv` for Vulkan. GPU tests require a device for the requested API. The upload test checks all three formats, both UV streams, odd counts, large offsets, negative/tiled coordinates, zero-width bounds, shared/overlapping mesh ranges, FP32 fallback, and skinning. The raster test checks mixed formats sharing a material and UNORM16 meshes sharing a buffer with different decode bounds in depth, forward, and GBuffer passes, including motion-vector variants. Without GPU arguments, CTest marks the raster test as skipped.
