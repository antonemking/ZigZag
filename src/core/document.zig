const std = @import("std");

pub const Document = struct {
    id: []const u8,
    content: []const u8,

    pub fn init(id: []const u8, content: []const u8) Document {
        return .{
            .id = id,
            .content = content,
        };
    }
};

test "document creation" {
    const doc = Document.init("doc1", "hello world");
    try std.testing.expectEqualStrings("doc1", doc.id);
    try std.testing.expectEqualStrings("hello world", doc.content);
}
