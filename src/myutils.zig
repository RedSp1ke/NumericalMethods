const std = @import("std");

fn does_contain(arr: []const u8, ch: u8) bool {
    for (arr) |el| {
        if (el == ch) {
            return true;
        }
    }
    return false;
}

pub fn read_until_any(wr: std.io.AnyReader, buf: []u8, seps: []const u8) anyerror!?[]u8 {
    var cur_ch: u8 = wr.readByte() catch return null;
    while (does_contain(seps, cur_ch)) {
        cur_ch = wr.readByte() catch return null;
    }

    var cur_len: usize = 0;
    while (!does_contain(seps, cur_ch) and cur_len < buf.len) {
        buf[cur_len] = cur_ch;
        cur_len += 1;
        cur_ch = wr.readByte() catch return buf[0..cur_len];
    }

    if (std.mem.containsAtLeast(u8, seps, 1, &.{cur_ch})) {
        return buf[0..cur_len];
    }

    return error.StreamTooLong;
}
