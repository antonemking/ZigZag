const std = @import("std");
const Document = @import("../core/document.zig").Document;

pub const BM25Index = struct {
    documents: std.ArrayList(Document),
    // Term -> [doc_ids that contain it]
    inverted_index: std.StringHashMap(std.ArrayList(usize)),
    // Term -> document frequency
    doc_frequencies: std.StringHashMap(usize),
    avg_doc_length: f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) BM25Index {
        return .{
            .documents = std.ArrayList(Document).init(allocator),
            .inverted_index = std.StringHashMap(std.ArrayList(usize)).init(allocator),
            .doc_frequencies = std.StringHashMap(usize).init(allocator),
            .avg_doc_length = 0.0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BM25Index) void {
        // Clean up inverted index lists
        var it = self.inverted_index.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinit();
        }
        self.documents.deinit();
        self.inverted_index.deinit();
        self.doc_frequencies.deinit();
    }

    pub fn addDocument(self: *BM25Index, doc: Document) !void {
        const doc_id = self.documents.items.len;
        try self.documents.append(doc);

        // Tokenize and index (simplified - just split on spaces)
        var tokens = std.mem.tokenizeScalar(u8, doc.content, ' ');
        while (tokens.next()) |token| {
            // Update inverted index
            const result = try self.inverted_index.getOrPut(token);
            if (!result.found_existing) {
                result.value_ptr.* = std.ArrayList(usize).init(self.allocator);
            }
            try result.value_ptr.append(doc_id);

            // Update document frequency
            const df_result = try self.doc_frequencies.getOrPut(token);
            if (!df_result.found_existing) {
                df_result.value_ptr.* = 1;
            } else {
                df_result.value_ptr.* += 1; // Increment count

            }
        }
    }
};

test "BM25 index basic operations" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var index = BM25Index.init(allocator);
    defer index.deinit();

    const doc1 = Document.init("doc1", "the quick brown fox");
    const doc2 = Document.init("doc2", "jumps over the lazy dog");
    try index.addDocument(doc1);
    try index.addDocument(doc2);

    try std.testing.expectEqual(index.documents.items.len, 2);
    const df_quick = index.doc_frequencies.get("quick");
    try std.testing.expect(df_quick.? == 1);
    const df_the = index.doc_frequencies.get("the");
    try std.testing.expect(df_the.? == 2);
}
