const std = @import("std");
const ut = @import("myutils.zig");
const mat = @import("matrix.zig");

fn gen_mat(alloc: std.mem.Allocator, size: usize, m: mat.FloatType) std.mem.Allocator.Error!mat.Matrix {
    var mt = try mat.Matrix.init(alloc, size);
    const q = 1.001 - 2 * m * 1e-3;

    for (0..size) |i| {
        for (0..size) |j| {
            if (i == j) {
                mt.arr[i][j] = std.math.pow(mat.FloatType, q - 1, @floatFromInt(i + j));
            } else {
                mt.arr[i][j] = std.math.pow(mat.FloatType, q, @floatFromInt(i + j)) + 0.1 * (@as(mat.FloatType, @floatFromInt(j)) - @as(mat.FloatType, @floatFromInt(i)));
            }
        }
    }

    return mt;
}

fn fill_vec(x: mat.FloatType, vec: []mat.FloatType) void {
    for (0..vec.len) |i| {
        const frac = @as(mat.FloatType, @floatFromInt(i + 1));
        vec[i] = x * @exp(frac) * @cos(frac);
    }
}

// better use buffered out stream instead of debug print
fn print_vec(vec: []mat.FloatType) void {
    std.debug.print("{d}", .{vec[0]});

    for (1..vec.len) |i| {
        std.debug.print(" & {d}", .{vec[i]});
    }
    std.debug.print("\n", .{});
}

pub fn main() !void {
    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);

    try bw.flush();

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();

    const test_file = (try std.fs.cwd().openFile("problem6.txt", .{})).reader();
    var br = std.io.bufferedReader(test_file);
    const fin = br.reader();

    const mtx = try mat.Matrix.init_reader(alloc, fin.any());
    mtx.debug_display();

    var vec_pure = try std.ArrayList(mat.FloatType).initCapacity(alloc, mtx.size);
    vec_pure.expandToCapacity();
    defer vec_pure.deinit();

    _ = try mat.read_vec(vec_pure.items, fin.any());
    var vecout = try vec_pure.clone();
    defer vecout.deinit();
    var vecin = try vec_pure.clone();
    defer vecin.deinit();

    std.debug.print("Matrix:\n", .{});
    for (0..mtx.size) |i| {
        std.debug.print("{d}", .{@as(i64, @intFromFloat(mtx.arr[i][0]))});

        for (1..mtx.size) |j| {
            std.debug.print(" & {d}", .{@as(i64, @intFromFloat(mtx.arr[i][j]))});
        }

        std.debug.print(" & {d} \\\\\n", .{vecin.items[i]});
    }

    const mtinv = try mtx.inv_al();
    std.debug.print("Invert matrix:\n", .{});
    for (0..mtinv.size) |i| {
        std.debug.print("{d}", .{mtinv.arr[i][0]});

        for (1..mtinv.size) |j| {
            std.debug.print(" & {d}", .{mtinv.arr[i][j]});
        }

        std.debug.print(" \\\\\n", .{});
    }

    std.debug.print("Determinant: {d}\n", .{try mtx.det_al()});
    std.debug.print("Condition N: {d}\n", .{try mtx.condition_al()});
    std.debug.print("Solution gauss dflt:\n", .{});
    print_vec(try mtx.gauss_al(vecout.items));

    vecin = try vec_pure.clone();
    const res = try mtx.gauss_main_al(vecin.items, vecout.items);
    std.debug.print("Solution gauss main:\n", .{});
    print_vec(res);
    std.debug.print("Mat * solution: {d}\n", .{try mtx.mulv_al(res)});
    @memcpy(vecin.items, vec_pure.items);

    var omega: mat.FloatType = 0.0;
    var min_it: usize = 5000;
    var min_omega: mat.FloatType = 0.0;
    while (omega < 2.0) : (omega += 0.001) {
        // std.debug.print("Upper relaxation: {d}\n", .{try mtx.upper_relaxation(vecin.items, res, 1.05, 1e-8)});
        var cur_it: usize = undefined;
        _ = try mtx.upper_relaxation(vecin.items, res, omega, 1e-11, &cur_it);

        if (cur_it < min_it) {
            min_it = cur_it;
            min_omega = omega;
        }
    }

    std.debug.print("Min it count: {d}\nfor omega: {d}\n\n", .{ min_it, min_omega });
    std.debug.print("Upper relaxation:\n", .{});
    print_vec(try mtx.upper_relaxation(vecin.items, res, min_omega, 1e-8, &min_it));
}
