__all__ = ["encode", "decode"]

import struct

import numba  # type: ignore
import numpy as np

_header_format_struct = struct.Struct(">IIBB")

RGBA = tuple[int, int, int, int]
RGB = tuple[int, int, int]

MAGIC_STRING = [int(c) for c in b"qoif"]

INDEX_DEFAULT_VALUE: RGBA = 0, 0, 0, 255
INDEX_SIZE: int = 64

STARTING_VALUE: RGBA = 0, 0, 0, 255

QOI_OP_RUN: int = 0b11000000
QOI_OP_DIFF: int = 0b01000000
QOI_OP_LUMA: int = 0b10000000
QOI_OP_RGB: int = 0b11111110
QOI_OP_RGBA: int = 0b11111111

END_STREAM_PILL = [int(c) for c in [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01]]

@numba.njit
def _encode_chunk(
        data: bytearray,
        qoi_index: bytearray,
        running_total: int,
        chunk_start: int,
        chunk_end: int,
        old_rgba: RGBA,
        write_index: int,
        write_buffer: bytearray,
        end_stream: bool = False
        ) -> tuple[int, int, RGBA]:
    
    for index in range(chunk_start, chunk_end, 4):
        r, g, b, a = data[index], data[index + 1], data[index + 2], data[index + 3]
        rgba = r, g, b, a

        if rgba == old_rgba:
            running_total += 1
            if running_total == 62:
                # QOI_OP_RUN max length hit
                write_buffer[write_index] = QOI_OP_RUN | 61 # bias = -1
                write_index += 1
                running_total = 0
            continue

        # rgba != old_rgba

        if running_total > 0:
            # QOI_OP_RUN
            write_buffer[write_index] = QOI_OP_RUN | (running_total - 1) # bias = -1
            write_index += 1
            running_total = 0

        index_position = ((3 * rgba[0] + 5 * rgba[1] + 7 * rgba[2] + 11 * rgba[3]) % 64)
        index_position_four = index_position * 4
        ir = qoi_index[index_position_four + 0]
        ig = qoi_index[index_position_four + 1]
        ib = qoi_index[index_position_four + 2]
        ia = qoi_index[index_position_four + 3]
        indexed = ir, ig, ib, ia
        dr, dg, db = (
            (rgba[0] - old_rgba[0] % 256), 
            (rgba[1] - old_rgba[1] % 256), 
            (rgba[2] - old_rgba[2] % 256)
            )
        dg_r = dr - dg
        dg_b = db - dg

        if indexed == rgba:
            write_buffer[write_index] = index_position
            write_index += 1

        elif (
                (dr > -3 and dr < 2)
                and (dg > -3 and dg < 2)
                and (db > -3 and db < 2)
                and (rgba[3] == old_rgba[3])
        ):
            # QOI_OP_DIFF
            write_buffer[write_index] = QOI_OP_DIFF | (dr + 2) << 4 | (dg + 2) << 2 | (db + 2)
            write_index += 1

        elif (
                (dg_r > -9 and dg_r < 8)
                and (dg > -33 and dg < 32)
                and (dg_b > -9 and dg_b < 8)
                and (rgba[3] == old_rgba[3])
            ):
            # QOI_OP_LUMA
            write_buffer[write_index] = QOI_OP_LUMA | (dg + 32)
            write_buffer[write_index + 1] = (dg_r + 8) << 4 | (dg_b + 8)
            write_index += 2

        elif rgba[3] == old_rgba[3]:
            write_buffer[write_index] = QOI_OP_RGB
            write_buffer[write_index + 1] = rgba[0]
            write_buffer[write_index + 2] = rgba[1]
            write_buffer[write_index + 3] = rgba[2]
            write_index += 4

        else:
            write_buffer[write_index] = QOI_OP_RGBA
            write_buffer[write_index + 1] = rgba[0]
            write_buffer[write_index + 2] = rgba[1]
            write_buffer[write_index + 3] = rgba[2]
            write_buffer[write_index + 4] = rgba[3]
            write_index += 5

        qoi_index[index_position_four + 0] = rgba[0]
        qoi_index[index_position_four + 1] = rgba[1]
        qoi_index[index_position_four + 2] = rgba[2]
        qoi_index[index_position_four + 3] = rgba[3]
        old_rgba = rgba

    if end_stream and running_total > 0:
        write_buffer[write_index] = QOI_OP_RUN | (running_total - 1)
        write_index += 1
        running_total = 0

    return write_index, running_total, old_rgba

@numba.njit
def _decode_chunk(
        data: np.ndarray,
        qoi_index: bytearray,
        chunk_start: int,
        chunk_end: int,
        old_rgba: RGBA,
        write_index: int,
        write_buffer: bytearray
        ) -> tuple[int, int, RGBA]:

    END_STREAM_PILL = 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01

    read_index = chunk_start
    
    while read_index < chunk_end:

        byte = data[read_index]

        if byte == 0x00:
            nextbyte = data[read_index + 1]
            if (nextbyte == 0x00):
                # two identical QOI_OP_INDEX codes consecutively is illegal 
                # so if the rest of the bytes doesn't equal the END_STREAM_PILL, 
                # we'll raise ValueError
                check_fail = False
                for i in range(8):
                    if data[read_index + i] != END_STREAM_PILL[i]:
                        check_fail = True
                if check_fail:
                    check_fail = False
                    for i in range(8):
                        if data[read_index + i + 1] != END_STREAM_PILL[i]:
                            check_fail = True
                if check_fail:
                    raise ValueError() # two consecutive QOI_OP_INDEX 0x00 hit
                else:
                    return write_index, -1, old_rgba

        if byte == QOI_OP_RGBA:
            # QOI_OP_RGBA
            write_buffer[write_index] = data[read_index + 1]
            write_buffer[write_index + 1] = data[read_index + 2]
            write_buffer[write_index + 2] = data[read_index + 3]
            write_buffer[write_index + 3] = data[read_index + 4]
            read_index += 5
            write_index += 4

        elif byte == QOI_OP_RGB:
            # QOI_OP_RGB
            write_buffer[write_index] = data[read_index + 1]
            write_buffer[write_index + 1] = data[read_index + 2]
            write_buffer[write_index + 2] = data[read_index + 3]
            write_buffer[write_index + 3] = old_rgba[3]
            read_index += 4
            write_index += 4

        elif byte >> 6 == 3:
            # QOI_OP_RUN
            count = (byte & 0b0011_1111) + 1 # bias = -1
            for _ in range(count):
                write_buffer[write_index] = old_rgba[0]
                write_buffer[write_index + 1] = old_rgba[1]
                write_buffer[write_index + 2] = old_rgba[2]
                write_buffer[write_index + 3] = old_rgba[3]
                write_index += 4
            read_index += 1
        
        elif byte >> 6 == 2:
            # QOI_OP_LUMA
            nextbyte = data[read_index + 1]
            dg = (byte & 0x3F) - 32
            write_buffer[write_index] = (old_rgba[0] + (dg - 8 + ((nextbyte >> 4) & 0x0F))) % 256
            write_buffer[write_index + 1] = (old_rgba[1] + (dg)) % 256
            write_buffer[write_index + 2] = (old_rgba[2] + (dg - 8 + (nextbyte & 0x0F))) % 256
            write_buffer[write_index + 3] = old_rgba[3]
            write_index += 4
            read_index += 2

        elif byte >> 6 == 1:
            # QOI_OP_DIFF
            write_buffer[write_index] = (old_rgba[0] + ((byte >> 4) & 0x03) - 2) % 256
            write_buffer[write_index + 1] = (old_rgba[1] + ((byte >> 2) & 0x03) - 2) % 256
            write_buffer[write_index + 2] = (old_rgba[2] + ((byte) & 0x03) - 2) % 256
            write_buffer[write_index + 3] = old_rgba[3]
            write_index += 4
            read_index += 1

        elif byte >> 6 == 0:
            # QOI_OP_INDEX
            index_index = 4 * byte
            write_buffer[write_index] = qoi_index[index_index]
            write_buffer[write_index + 1] = qoi_index[index_index + 1]
            write_buffer[write_index + 2] = qoi_index[index_index + 2]
            write_buffer[write_index + 3] = qoi_index[index_index + 3]
            write_index += 4
            read_index += 1

        else:
            assert False

        r, g, b, a = write_buffer[write_index - 4], write_buffer[write_index - 3], write_buffer[write_index - 2], write_buffer[write_index - 1]
        hash_ = ((3 * r + 5 * g + 7 * b + 11 * a) % 64) * 4
        qoi_index[hash_] = r
        qoi_index[hash_ + 1] = g
        qoi_index[hash_ + 2] = b
        qoi_index[hash_ + 3] = a
        old_rgba = r, g, b, a  

    return write_index, read_index, old_rgba 

def decode(
        data
        ) -> np.ndarray:

    header = bytes(data[4:14:])
    width, height, channels, colorspace = _header_format_struct.unpack(header)

    write_index = 0
    read_index = 14
    decoded = np.empty(len(data) * 65 * 4, dtype=np.uint8)
    old_rgba: tuple[int, int, int, int] = (0, 0, 0, 255)
    qoi_index = bytearray(256)
    qoi_index[3::4] = (255 for _ in range(64))

    write_index, read_index, old_rgba = _decode_chunk(
        data, qoi_index, read_index, len(data), old_rgba, write_index, decoded
    )

    return decoded[:write_index:]

def encode(
        data, 
        width: int,
        height: int,
        size: tuple[int, int] = None,
        colorspace: int = 0,
        ) -> np.ndarray:

    if size is not None:
        width, height = size
    if not (isinstance(width, int) and isinstance(height, int)):
        raise ValueError("Must provide width, height (or size)")
    
    header = [int(c) for c in _header_format_struct.pack(width, height, 4, colorspace)]
    
    encoded = np.empty(len(data) * 5 + 22, dtype=np.uint8)
    qoi_index = np.zeros(256, dtype=np.uint8)
    qoi_index[3::4] = [255 for _ in range(64)]

    write_index, _, _ = _encode_chunk(
        data, qoi_index, 0, 0, len(data), (0, 0, 0, 255), 14, encoded, True
    )

    encoded[:4:] = MAGIC_STRING
    encoded[4:14:] = header
    encoded[write_index:write_index + 8] = END_STREAM_PILL
    return encoded[:write_index + 8:]
