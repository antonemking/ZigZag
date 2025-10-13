const std = @import("std");

pub const SearchResult = struct {
    doc_id: []const u8,
    score: f32,

    pub fn init(doc_id: []const u8, score: f32) SearchResult {
        return .{
            .doc_id = doc_id,
            .score = score,
        };
    }

    // For sorting results by score (descending)
    pub fn compareDesc(_: void, a: SearchResult, b: SearchResult) bool {
        return a.score > b.score;
    }
};

pub const RankedResults = struct {
    results: []SearchResult,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, capacity: usize) !RankedResults {
        const results = try allocator.alloc(SearchResult, capacity);
        return .{
            .results = results,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RankedResults) void {
        self.allocator.free(self.results);
    }

    pub fn sort(self: *RankedResults) void {
        std.sort.block(SearchResult, self.results, {}, SearchResult.compareDesc);
    }
};

test "ranked results sorting" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var ranked = try RankedResults.init(allocator, 3);
    defer ranked.deinit();

    ranked.results[0] = SearchResult.init("doc1", 0.5);
    ranked.results[1] = SearchResult.init("doc2", 0.9);
    ranked.results[2] = SearchResult.init("doc3", 0.3);

    ranked.sort();

    try std.testing.expectEqual(@as(f32, 0.9), ranked.results[0].score);
    try std.testing.expectEqualStrings("doc2", ranked.results[0].doc_id);
}
