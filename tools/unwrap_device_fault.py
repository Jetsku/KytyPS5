"""Extract vendor crash data from a Vulkan device-fault binary header.

The emulator's _device_fault.nv-gpudmp includes the Vulkan header; an NVIDIA
Aftermath reader expects the vendor payload after that header instead.
"""

import argparse
import json
import struct
from pathlib import Path


def unwrap(data: bytes):
    header = struct.Struct("<5I16s5I")
    if len(data) < header.size:
        raise ValueError("truncated Vulkan device-fault header")
    size, version, vendor, device, driver, uuid, app, app_version, engine, engine_version, api = header.unpack_from(data)
    if version != 1 or size < header.size or size >= len(data):
        raise ValueError("not a supported Vulkan device-fault vendor binary")
    return {
        "header_bytes": size,
        "header_version": version,
        "vendor_id": hex(vendor),
        "device_id": hex(device),
        "driver_version_raw": hex(driver),
        "pipeline_cache_uuid": uuid.hex(),
        "application_name_offset": app,
        "application_version": app_version,
        "engine_name_offset": engine,
        "engine_version": engine_version,
        "api_version_raw": hex(api),
        "payload_bytes": len(data) - size,
    }, data[size:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="new vendor-payload file (input is preserved)")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    try:
        metadata, payload = unwrap(args.input.read_bytes())
        with args.output.open("xb") as file:
            file.write(payload)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
