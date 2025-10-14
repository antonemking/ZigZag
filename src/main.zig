const std = @import("std");

// Import modules so their tests are included
const document = @import("core/document.zig");
const result = @import("core/result.zig");
const bm25_scorer = @import("bm25/scorer.zig");
const bm25_index = @import("bm25/index.zig");
const onnx = @import("rerank/onnx.zig");

pub fn main() !void {
    std.debug.print("ZigZag - Fast Reranking\n", .{});
}

// Expose tests from imported modules
test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(document);
    std.testing.refAllDecls(result);
    std.testing.refAllDecls(bm25_scorer);
    std.testing.refAllDecls(bm25_index);
    std.testing.refAllDecls(onnx);
}
