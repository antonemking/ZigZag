const std = @import("std");

// Import modules so their tests are included
const document = @import("core/document.zig");
const result = @import("core/result.zig");

pub fn main() !void {
    std.debug.print("ZigZag - Fast Reranking\n", .{});
}

// Expose tests from imported modules
test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(document);
    std.testing.refAllDecls(result);
}
