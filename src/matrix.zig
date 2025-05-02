const std = @import("std");
const ut = @import("myutils.zig");

pub const FloatType = f64;
pub const EPS: FloatType = 1e-8;
pub const Error = error{
    VecDoesntFit,
    MatDoesntFit,
    NotEnoughInput,
    IncorrectInput,
    DegenerateMatrix,
};

pub fn vec_norm(vec: []FloatType) FloatType {
    var res = @abs(vec[0]);
    for (1..vec.len) |i| {
        res = @max(res, @abs(vec[i]));
    }

    return res;
}

pub fn vec_norm_sub(a: []FloatType, b: []FloatType) FloatType {
    var res = @abs(a[0] - b[0]);
    for (1..a.len) |i| {
        res = @max(res, @abs(a[i] - b[i]));
    }

    return res;
}

pub fn read_vec(vec: []FloatType, inp: std.io.AnyReader) anyerror![]FloatType {
    const seps = "\t\n\r ";
    var buf: [32]u8 = undefined;

    for (0..vec.len) |i| {
        const num_str = try ut.read_until_any(inp, &buf, seps) orelse return Error.NotEnoughInput;
        vec[i] = try std.fmt.parseFloat(FloatType, num_str);
    }

    return vec;
}

pub const Matrix = struct {
    const Self = @This();

    alloc: std.mem.Allocator,

    arr: [][]FloatType,
    size: usize,

    fn fits(self: Self, vec: []const FloatType) bool {
        return self.size == vec.len;
    }

    fn fitsm(self: Self, mat: Matrix) bool {
        return self.size == mat.size;
    }

    pub fn init(m_alloc: std.mem.Allocator, m_size: usize) std.mem.Allocator.Error!Self {
        var arr = try m_alloc.alloc([]FloatType, m_size);
        for (0..m_size) |i| {
            arr[i] = try m_alloc.alloc(FloatType, m_size);
        }

        return Self{
            .alloc = m_alloc,
            .size = m_size,
            .arr = arr,
        };
    }

    pub fn deinit(self: Self) void {
        for (0..self.size) |i| {
            self.alloc.free(self.arr[i]);
        }
        self.alloc.free(self.arr);
    }

    pub fn debug_display(self: Self) void {
        for (0..self.size) |i| {
            for (0..self.size) |j| {
                std.debug.print("{d}\t", .{self.arr[i][j]});
            }
            std.debug.print("\n", .{});
        }
    }

    pub fn mulv(self: Self, in: []const FloatType, out: []FloatType) Error![]FloatType {
        if (!self.fits(in) or !self.fits(out)) {
            return Error.VecDoesntFit;
        }

        for (0..self.size) |i| {
            out[i] = 0;
            for (0..self.size) |j| {
                out[i] += self.arr[i][j] * in[j];
            }
        }

        return out;
    }

    pub fn mulv_al(self: Self, in: []const FloatType) anyerror![]FloatType {
        const out: []FloatType = try self.alloc.alloc(FloatType, self.size);
        return self.mulv(in, out);
    }

    pub fn identity(self: Self) Self {
        for (0..self.size) |i| {
            @memset(self.arr[i], 0.0);
            self.arr[i][i] = 1;
        }
        return self;
    }

    pub fn zero(self: Self) Self {
        for (0..self.size) |i| {
            @memset(self.arr[i], 0.0);
        }
        return self;
    }

    pub fn from_reader(self: Self, inp: std.io.AnyReader) anyerror!Self {
        const seps = "\t\n\r ";
        var buf: [32]u8 = undefined;

        for (0..self.size) |i| {
            for (0..self.size) |j| {
                const num_str = try ut.read_until_any(inp, &buf, seps) orelse return Error.NotEnoughInput;
                self.arr[i][j] = try std.fmt.parseFloat(FloatType, num_str);
            }
        }

        return self;
    }

    pub fn init_reader(m_alloc: std.mem.Allocator, inp: std.io.AnyReader) anyerror!Self {
        const seps = "\t\n\r ";
        var buf: [32]u8 = undefined;

        const num_str = try ut.read_until_any(inp, &buf, seps) orelse return Error.NotEnoughInput;
        const m_size = try std.fmt.parseInt(usize, num_str, 0);

        return (try Self.init(m_alloc, m_size)).from_reader(inp);
    }

    pub fn copy(self: Self, src: Matrix) Error!Self {
        if (!self.fitsm(src)) {
            return Error.MatDoesntFit;
        }
        for (0..self.size) |i| {
            @memcpy(self.arr[i], src.arr[i]);
        }

        return self;
    }

    pub fn clone(self: Self, n_alloc: std.mem.Allocator) std.mem.Allocator.Error!Self {
        return (try Self.init(n_alloc, self.size)).copy(self) catch unreachable;
    }

    // in vector stores the right part vector and
    // used to store solution
    pub fn gauss(self: Self, in: []FloatType) Error![]FloatType {
        if (!self.fits(in)) {
            return Error.VecDoesntFit;
        }
        var cur_val: FloatType = undefined;

        for (0..self.size) |i| {
            for (i..self.size) |j| {
                if (@abs(self.arr[j][i]) > EPS) {
                    std.mem.swap(@TypeOf(self.arr[0]), &self.arr[i], &self.arr[j]);
                    std.mem.swap(FloatType, &in[i], &in[j]);

                    break;
                }
            }

            if (@abs(self.arr[i][i]) < EPS) {
                return Error.DegenerateMatrix;
            }

            cur_val = self.arr[i][i];

            for (i..self.size) |j| {
                self.arr[i][j] /= cur_val;
            }
            in[i] /= cur_val;

            for ((i + 1)..self.size) |j| {
                cur_val = self.arr[j][i];

                for ((i)..self.size) |colomn| {
                    self.arr[j][colomn] -= self.arr[i][colomn] * cur_val;
                }

                in[j] -= in[i] * cur_val;
            }
        }

        var i = self.size - 1;
        while (i > 0) : (i -%= 1) {
            for ((i +% 1)..self.size) |j| {
                in[i] -= self.arr[i][j] * in[j];
            }
        }
        for (1..self.size) |j| {
            in[0] -= self.arr[0][j] * in[j];
        }

        return in;
    }

    pub fn gauss_al(self: Self, in: []FloatType) anyerror![]FloatType {
        var tmp = try self.clone(self.alloc);
        defer tmp.deinit();
        return tmp.gauss(in);
    }

    fn swap_column(self: Self, col1: usize, col2: usize) Self {
        for (0..self.size) |i| {
            std.mem.swap(FloatType, &self.arr[i][col1], &self.arr[i][col2]);
        }

        return self;
    }

    pub fn gauss_main(self: Self, in: []FloatType, out: []FloatType) Error![]FloatType {
        if (!self.fits(in) and !self.fits(out)) {
            return Error.VecDoesntFit;
        }
        var cur_val: FloatType = undefined;

        for (0..self.size) |i| {
            out[i] = @floatFromInt(i);
        }
        // std.debug.print("[DEBUG] matrix:\n", .{});
        // self.debug_display();
        for (0..self.size) |i| {
            var max: FloatType = 0.0;
            var i_max: usize = 0;
            for (i..self.size) |j| {
                const cur_abs = @abs(self.arr[i][j]);
                if (cur_abs > max) {
                    max = cur_abs;
                    i_max = j;
                }
            }

            if (max < EPS) {
                return Error.DegenerateMatrix;
            }

            if (i != i_max) {
                _ = self.swap_column(i, i_max);
                std.mem.swap(FloatType, &out[i], &out[i_max]);
            }
            // std.debug.print("[DEBUG] matrix:\n", .{});
            // self.debug_display();

            cur_val = self.arr[i][i];

            for (i..self.size) |j| {
                self.arr[i][j] /= cur_val;
            }
            in[i] /= cur_val;

            for ((i + 1)..self.size) |j| {
                cur_val = self.arr[j][i];

                for ((i)..self.size) |colomn| {
                    self.arr[j][colomn] -= self.arr[i][colomn] * cur_val;
                }

                in[j] -= in[i] * cur_val;
            }
        }

        var i = self.size - 1;
        while (i > 0) : (i -%= 1) {
            for ((i +% 1)..self.size) |j| {
                in[i] -= self.arr[i][j] * in[j];
            }
        }
        for (1..self.size) |j| {
            in[0] -= self.arr[0][j] * in[j];
        }

        for (0..self.size) |j| {
            out[j] = in[@intFromFloat(out[j])];
        }

        return out;
    }

    pub fn gauss_main_al(self: Self, in: []FloatType, out: []FloatType) anyerror![]FloatType {
        var tmp = try self.clone(self.alloc);
        defer tmp.deinit();
        return tmp.gauss_main(in, out);
    }

    pub fn det(self: Self) FloatType {
        var cur_val: FloatType = undefined;
        var res: FloatType = 1;

        for (0..self.size) |i| {
            for (i..self.size) |j| {
                if (@abs(self.arr[j][i]) > EPS) {
                    if (j == i) {
                        break;
                    }

                    std.mem.swap(@TypeOf(self.arr[0]), &self.arr[i], &self.arr[j]);
                    res *= -1;
                    break;
                }
            }

            if (@abs(self.arr[i][i]) < EPS) {
                return 0;
            }

            cur_val = self.arr[i][i];
            res *= cur_val;

            for (i..self.size) |j| {
                self.arr[i][j] /= cur_val;
            }

            for ((i + 1)..self.size) |j| {
                cur_val = self.arr[j][i];

                for ((i)..self.size) |colomn| {
                    self.arr[j][colomn] -= self.arr[i][colomn] * cur_val;
                }
            }
        }

        return res;
    }

    pub fn det_al(self: Self) std.mem.Allocator.Error!FloatType {
        const tmp = try self.clone(self.alloc);
        defer tmp.deinit();
        return tmp.det();
    }

    // multiplies out by inverted matrix of self
    // self becomes identity matrix
    pub fn inv(self: Self, out: Self) Error!Self {
        if (!self.fitsm(out)) {
            return Error.MatDoesntFit;
        }
        var cur_val: FloatType = undefined;

        for (0..self.size) |i| {
            for (i..self.size) |j| {
                if (@abs(self.arr[j][i]) > EPS) {
                    if (j == i) {
                        break;
                    }

                    std.mem.swap(@TypeOf(self.arr[0]), &self.arr[i], &self.arr[j]);
                    std.mem.swap(@TypeOf(out.arr[0]), &out.arr[i], &out.arr[j]);

                    break;
                }
            }

            if (@abs(self.arr[i][i]) < EPS) {
                return Error.DegenerateMatrix;
            }

            cur_val = self.arr[i][i];

            for (0..self.size) |j| {
                self.arr[i][j] /= cur_val;
                out.arr[i][j] /= cur_val;
            }

            for ((i + 1)..self.size) |j| {
                cur_val = self.arr[j][i];

                for (0..self.size) |colomn| {
                    self.arr[j][colomn] -= self.arr[i][colomn] * cur_val;
                    out.arr[j][colomn] -= out.arr[i][colomn] * cur_val;
                }
            }
        }

        for (0..self.size) |i_inv| {
            const i = self.size - i_inv - 1;

            for (0..i) |j| {
                cur_val = self.arr[j][i];

                for (0..self.size) |colomn| {
                    self.arr[j][colomn] -= self.arr[i][colomn] * cur_val;
                    out.arr[j][colomn] -= out.arr[i][colomn] * cur_val;
                }
            }
        }
        return out;
    }

    pub fn inv_al(self: Self) anyerror!Self {
        const out = (try Self.init(self.alloc, self.size)).identity();
        const tmp = try self.clone(self.alloc);
        defer tmp.deinit();
        return tmp.inv(out);
    }

    // ||A||<inf> norm - convinient and fast to calculate
    pub fn norm(self: Self) FloatType {
        var res: FloatType = 0;
        for (0..self.size) |i| {
            var sum: FloatType = 0;
            for (0..self.size) |j| {
                sum += @abs(self.arr[i][j]);
            }

            res = @max(res, sum);
        }

        return res;
    }

    pub fn condition_al(self: Self) anyerror!FloatType {
        const tmp = try self.inv_al();
        defer tmp.deinit();
        return self.norm() * tmp.norm();
    }

    fn calc_upper_relaxation(self: Self, in: []FloatType, prev: []FloatType, out: []FloatType, omega: FloatType) void {
        for (0..self.size) |i| {
            var sum_prev: FloatType = 0;
            for (0..i) |j| {
                sum_prev += self.arr[i][j] * out[j];
            }

            var sum_curr: FloatType = 0;
            for (i..self.size) |j| {
                sum_curr += self.arr[i][j] * prev[j];
            }

            out[i] = prev[i] + omega / self.arr[i][i] * (in[i] - sum_prev - sum_curr);
        }
    }

    pub fn upper_relaxation(self: Self, in: []FloatType, out: []FloatType, omega: FloatType, precision: FloatType, its: *usize) anyerror![]FloatType {
        if (!self.fits(in) or !self.fits(out)) {
            return Error.VecDoesntFit;
        }

        const tmp: []FloatType = try self.alloc.alloc(FloatType, self.size);
        defer self.alloc.free(tmp);

        for (0..self.size) |i| {
            out[i] = 0;
        }

        var it_count: usize = 0;
        while (vec_norm_sub(in, self.mulv(out, tmp) catch unreachable) > precision) {
            if (it_count > 500) {
                // std.debug.print("Too many iterations: {d}\n", .{omega});
                break;
            }

            @memcpy(tmp, out);
            self.calc_upper_relaxation(in, tmp, out, omega);
            // std.debug.print("[DEBUG:{d}]\tnew out vec is {any}\n", .{ it_count, out });

            it_count += 1;
        }

        its.* = it_count;

        return out;
    }
};
