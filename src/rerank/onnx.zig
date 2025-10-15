const std = @import("std");

// ONNX Runtime C API bindings
const c = @cImport({
    @cInclude("onnxruntime/onnxruntime_c_api.h");
});

pub const ONNXError = error{
    InitializationFailed,
    SessionCreationFailed,
    InferenceFailed,
    InvalidInput,
    OnnxRuntimeNotFound,
};

// ONNX Runtime session wrapping a loaded model
pub const ONNXSession = struct {
    env: ?*c.OrtEnv,
    session: ?*c.OrtSession,
    allocator: std.mem.Allocator,
    api: ?*const c.OrtApi,

    // Load a cross-encoder model from disk
    // model_path: path to .onnx file (e.g., "models/minilm.onnx")
    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !ONNXSession {
        // Get the ONNX Runtime C API
        const api_base = c.OrtGetApiBase();
        if (api_base == null) return ONNXError.OnnxRuntimeNotFound;

        const api = api_base.*.GetApi.?(c.ORT_API_VERSION);
        if (api == null) return ONNXError.OnnxRuntimeNotFound;

        // Create ONNX environment
        var env: ?*c.OrtEnv = null;
        var status = api.*.CreateEnv.?(c.ORT_LOGGING_LEVEL_WARNING, "ZigZag", &env);
        if (status != null) {
            api.*.ReleaseStatus.?(status);
            return ONNXError.InitializationFailed;
        }

        // Create session options with performance optimizations
        var session_options: ?*c.OrtSessionOptions = null;
        status = api.*.CreateSessionOptions.?(&session_options);
        if (status != null) {
            api.*.ReleaseStatus.?(status);
            api.*.ReleaseEnv.?(env);
            return ONNXError.InitializationFailed;
        }
        defer api.*.ReleaseSessionOptions.?(session_options);

        // Enable graph optimization for maximum performance
        status = api.*.SetSessionGraphOptimizationLevel.?(session_options, c.ORT_ENABLE_ALL);
        if (status != null) {
            api.*.ReleaseStatus.?(status);
        }

        // Try to enable CUDA execution provider for GPU acceleration
        // Set USE_GPU=1 environment variable to enable GPU
        const use_gpu = std.process.hasEnvVarConstant("USE_GPU");
        if (use_gpu) {
            // Attempt to append CUDA execution provider
            // This requires onnxruntime-gpu package to be installed
            status = api.*.SessionOptionsAppendExecutionProvider_CUDA_V2.?(session_options, null);
            if (status != null) {
                api.*.ReleaseStatus.?(status);
                std.debug.print("Warning: CUDA GPU not available, falling back to CPU\n", .{});
            } else {
                std.debug.print("Using CUDA GPU acceleration\n", .{});
            }
        }

        // Set CPU threading options (always set, used if GPU not available)
        status = api.*.SetIntraOpNumThreads.?(session_options, 0); // 0 = use all cores
        if (status != null) {
            api.*.ReleaseStatus.?(status);
        }

        status = api.*.SetInterOpNumThreads.?(session_options, 0); // 0 = use all cores
        if (status != null) {
            api.*.ReleaseStatus.?(status);
        }

        // Convert path to null-terminated for C API
        const path_z = try allocator.dupeZ(u8, model_path);
        defer allocator.free(path_z);

        // Load the model
        var session: ?*c.OrtSession = null;
        status = api.*.CreateSession.?(env, path_z.ptr, session_options, &session);
        if (status != null) {
            api.*.ReleaseStatus.?(status);
            api.*.ReleaseEnv.?(env);
            return ONNXError.SessionCreationFailed;
        }

        return ONNXSession{
            .env = env,
            .session = session,
            .allocator = allocator,
            .api = api,
        };
    }

    pub fn deinit(self: *ONNXSession) void {
        if (self.api) |api| {
            if (self.session) |session| {
                api.ReleaseSession.?(session);
            }
            if (self.env) |env| {
                api.ReleaseEnv.?(env);
            }
        }
    }

    // Score a single (query, document) pair
    // input_ids: tokenized text as integers (e.g., [101, 2023, 2003, ...])
    // attention_mask: which tokens to pay attention to (usually all 1s)
    // Returns: relevance score (higher = more relevant)
    pub fn infer(
        self: *ONNXSession,
        input_ids: []const i64,
        attention_mask: []const i64,
    ) !f32 {
        // Create token_type_ids (all zeros for single-sequence input)
        const token_type_ids = try self.allocator.alloc(i64, input_ids.len);
        defer self.allocator.free(token_type_ids);
        @memset(token_type_ids, 0);

        return self.inferWithTokenTypes(input_ids, attention_mask, token_type_ids);
    }

    // Internal inference with token_type_ids
    fn inferWithTokenTypes(
        self: *ONNXSession,
        input_ids: []const i64,
        attention_mask: []const i64,
        token_type_ids: []const i64,
    ) !f32 {
        const api = self.api orelse return ONNXError.OnnxRuntimeNotFound;

        if (input_ids.len != attention_mask.len) return ONNXError.InvalidInput;
        const seq_len = input_ids.len;

        // Create memory info for CPU
        var memory_info: ?*c.OrtMemoryInfo = null;
        var status = api.CreateCpuMemoryInfo.?(c.OrtArenaAllocator, c.OrtMemTypeDefault, &memory_info);
        if (status != null) {
            api.ReleaseStatus.?(status);
            return ONNXError.InferenceFailed;
        }
        defer api.ReleaseMemoryInfo.?(memory_info);

        // Create input tensors - shape is [1, seq_len] (batch_size=1)
        const input_shape = [_]i64{ 1, @intCast(seq_len) };

        // Create input_ids tensor
        var input_ids_tensor: ?*c.OrtValue = null;
        status = api.CreateTensorWithDataAsOrtValue.?(
            memory_info,
            @constCast(input_ids.ptr),
            input_ids.len * @sizeOf(i64),
            &input_shape,
            input_shape.len,
            c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &input_ids_tensor,
        );
        if (status != null) {
            api.ReleaseStatus.?(status);
            return ONNXError.InferenceFailed;
        }
        defer api.ReleaseValue.?(input_ids_tensor);

        // Create attention_mask tensor
        var attention_mask_tensor: ?*c.OrtValue = null;
        status = api.CreateTensorWithDataAsOrtValue.?(
            memory_info,
            @constCast(attention_mask.ptr),
            attention_mask.len * @sizeOf(i64),
            &input_shape,
            input_shape.len,
            c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &attention_mask_tensor,
        );
        if (status != null) {
            api.ReleaseStatus.?(status);
            return ONNXError.InferenceFailed;
        }
        defer api.ReleaseValue.?(attention_mask_tensor);

        // Create token_type_ids tensor
        var token_type_ids_tensor: ?*c.OrtValue = null;
        status = api.CreateTensorWithDataAsOrtValue.?(
            memory_info,
            @constCast(token_type_ids.ptr),
            token_type_ids.len * @sizeOf(i64),
            &input_shape,
            input_shape.len,
            c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &token_type_ids_tensor,
        );
        if (status != null) {
            api.ReleaseStatus.?(status);
            return ONNXError.InferenceFailed;
        }
        defer api.ReleaseValue.?(token_type_ids_tensor);

        // Run inference with all three inputs
        const input_names = [_][*c]const u8{ "input_ids", "attention_mask", "token_type_ids" };
        const inputs = [_]?*c.OrtValue{ input_ids_tensor, attention_mask_tensor, token_type_ids_tensor };
        const output_names = [_][*c]const u8{"logits"};
        var outputs: [1]?*c.OrtValue = [_]?*c.OrtValue{null};

        status = api.Run.?(
            self.session,
            null, // run options
            &input_names,
            &inputs,
            inputs.len,
            &output_names,
            output_names.len,
            &outputs,
        );
        if (status != null) {
            api.ReleaseStatus.?(status);
            return ONNXError.InferenceFailed;
        }
        defer if (outputs[0]) |output| api.ReleaseValue.?(output);

        // Extract the output score
        var output_data: ?*anyopaque = null;
        status = api.GetTensorMutableData.?(outputs[0], &output_data);
        if (status != null) {
            api.ReleaseStatus.?(status);
            return ONNXError.InferenceFailed;
        }

        // Output is a single float (cross-encoder score)
        const score_ptr: [*]f32 = @ptrCast(@alignCast(output_data));
        return score_ptr[0];
    }

    // Score multiple (query, doc) pairs at once - TRUE BATCHED INFERENCE
    // This is the fast path that gives us 10-100x speedup over sequential
    // batch_input_ids: array of tokenized inputs (each same length)
    // batch_attention_mask: array of attention masks (each same length)
    // Returns: scores for each pair (caller must free)
    pub fn inferBatch(
        self: *ONNXSession,
        batch_input_ids: []const []const i64,
        batch_attention_mask: []const []const i64,
    ) ![]f32 {
        const api = self.api orelse return ONNXError.OnnxRuntimeNotFound;

        const batch_size = batch_input_ids.len;
        if (batch_size == 0) return ONNXError.InvalidInput;
        if (batch_attention_mask.len != batch_size) return ONNXError.InvalidInput;

        const seq_len = batch_input_ids[0].len;
        if (seq_len == 0) return ONNXError.InvalidInput;

        // Pack all sequences into contiguous buffers for batched inference
        // Shape: [batch_size, seq_len] flattened to [batch_size * seq_len]
        const total_elements = batch_size * seq_len;
        const packed_input_ids = try self.allocator.alloc(i64, total_elements);
        defer self.allocator.free(packed_input_ids);
        const packed_attention_mask = try self.allocator.alloc(i64, total_elements);
        defer self.allocator.free(packed_attention_mask);
        const packed_token_type_ids = try self.allocator.alloc(i64, total_elements);
        defer self.allocator.free(packed_token_type_ids);

        // Copy data into packed buffers (row-major order)
        for (batch_input_ids, batch_attention_mask, 0..) |input_ids, attention_mask, batch_idx| {
            if (input_ids.len != seq_len) return ONNXError.InvalidInput;
            if (attention_mask.len != seq_len) return ONNXError.InvalidInput;

            const offset = batch_idx * seq_len;
            @memcpy(packed_input_ids[offset .. offset + seq_len], input_ids);
            @memcpy(packed_attention_mask[offset .. offset + seq_len], attention_mask);
            @memset(packed_token_type_ids[offset .. offset + seq_len], 0); // all zeros
        }

        // Create memory info for CPU
        var memory_info: ?*c.OrtMemoryInfo = null;
        var status = api.CreateCpuMemoryInfo.?(c.OrtArenaAllocator, c.OrtMemTypeDefault, &memory_info);
        if (status != null) {
            api.ReleaseStatus.?(status);
            return ONNXError.InferenceFailed;
        }
        defer api.ReleaseMemoryInfo.?(memory_info);

        // Create batched tensors - shape is [batch_size, seq_len]
        const input_shape = [_]i64{ @intCast(batch_size), @intCast(seq_len) };

        // Create input_ids tensor
        var input_ids_tensor: ?*c.OrtValue = null;
        status = api.CreateTensorWithDataAsOrtValue.?(
            memory_info,
            packed_input_ids.ptr,
            total_elements * @sizeOf(i64),
            &input_shape,
            input_shape.len,
            c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &input_ids_tensor,
        );
        if (status != null) {
            api.ReleaseStatus.?(status);
            return ONNXError.InferenceFailed;
        }
        defer api.ReleaseValue.?(input_ids_tensor);

        // Create attention_mask tensor
        var attention_mask_tensor: ?*c.OrtValue = null;
        status = api.CreateTensorWithDataAsOrtValue.?(
            memory_info,
            packed_attention_mask.ptr,
            total_elements * @sizeOf(i64),
            &input_shape,
            input_shape.len,
            c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &attention_mask_tensor,
        );
        if (status != null) {
            api.ReleaseStatus.?(status);
            return ONNXError.InferenceFailed;
        }
        defer api.ReleaseValue.?(attention_mask_tensor);

        // Create token_type_ids tensor
        var token_type_ids_tensor: ?*c.OrtValue = null;
        status = api.CreateTensorWithDataAsOrtValue.?(
            memory_info,
            packed_token_type_ids.ptr,
            total_elements * @sizeOf(i64),
            &input_shape,
            input_shape.len,
            c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &token_type_ids_tensor,
        );
        if (status != null) {
            api.ReleaseStatus.?(status);
            return ONNXError.InferenceFailed;
        }
        defer api.ReleaseValue.?(token_type_ids_tensor);

        // Run batched inference (single model call!)
        const input_names = [_][*c]const u8{ "input_ids", "attention_mask", "token_type_ids" };
        const inputs = [_]?*c.OrtValue{ input_ids_tensor, attention_mask_tensor, token_type_ids_tensor };
        const output_names = [_][*c]const u8{"logits"};
        var outputs: [1]?*c.OrtValue = [_]?*c.OrtValue{null};

        status = api.Run.?(
            self.session,
            null, // run options
            &input_names,
            &inputs,
            inputs.len,
            &output_names,
            output_names.len,
            &outputs,
        );
        if (status != null) {
            api.ReleaseStatus.?(status);
            return ONNXError.InferenceFailed;
        }
        defer if (outputs[0]) |output| api.ReleaseValue.?(output);

        // Check output tensor shape
        var type_info: ?*c.OrtTensorTypeAndShapeInfo = null;
        status = api.GetTensorTypeAndShape.?(outputs[0], &type_info);
        if (status != null) {
            api.ReleaseStatus.?(status);
            return ONNXError.InferenceFailed;
        }
        defer api.ReleaseTensorTypeAndShapeInfo.?(type_info);

        var num_dims: usize = undefined;
        status = api.GetDimensionsCount.?(type_info, &num_dims);
        if (status != null) {
            api.ReleaseStatus.?(status);
            return ONNXError.InferenceFailed;
        }

        // Extract batched output scores
        var output_data: ?*anyopaque = null;
        status = api.GetTensorMutableData.?(outputs[0], &output_data);
        if (status != null) {
            api.ReleaseStatus.?(status);
            return ONNXError.InferenceFailed;
        }

        // Copy all scores to result array
        // Output shape is typically [batch_size, 1] for cross-encoders
        const score_ptr: [*]f32 = @ptrCast(@alignCast(output_data));
        const scores = try self.allocator.alloc(f32, batch_size);

        // If output is [batch_size, 1], we need to extract every element
        // If output is [batch_size], we copy directly
        if (num_dims == 2) {
            // Shape is [batch_size, 1], extract first element of each row
            for (0..batch_size) |i| {
                scores[i] = score_ptr[i];
            }
        } else {
            // Shape is [batch_size], copy directly
            for (0..batch_size) |i| {
                scores[i] = score_ptr[i];
            }
        }

        return scores;
    }
};

test "Load real MiniLM model" {
    const allocator = std.testing.allocator;

    // Load the actual downloaded model
    var session = try ONNXSession.init(allocator, "models/minilm.onnx");
    defer session.deinit();

    // Verify session and env were created
    try std.testing.expect(session.session != null);
    try std.testing.expect(session.env != null);
    try std.testing.expect(session.api != null);
}

test "ONNX real inference" {
    const allocator = std.testing.allocator;

    var session = try ONNXSession.init(allocator, "models/minilm.onnx");
    defer session.deinit();

    // Real tokenized input (BERT-style token IDs)
    const input_ids = [_]i64{ 101, 2023, 2003, 1037, 3231, 102 };
    const attention_mask = [_]i64{ 1, 1, 1, 1, 1, 1 };

    const score = try session.infer(&input_ids, &attention_mask);

    // Real inference should return a valid score (not necessarily 0.0)
    // Cross-encoder scores are typically floats
    std.debug.print("\nGot cross-encoder score: {d}\n", .{score});
    try std.testing.expect(!std.math.isNan(score));
}

test "ONNX inference with padded sequences" {
    const allocator = std.testing.allocator;

    var session = try ONNXSession.init(allocator, "models/minilm.onnx");
    defer session.deinit();

    // First test pair from test_data.json: "how to make pasta" + "Pasta recipe: Boil water, add salt..."
    // This should score high (relevant match)
    const relevant_input_ids = [_]i64{
        101,  2129, 2000, 2191, 24857, 102,  24857, 17974, 1024, 26077, 2300, 1010, 5587, 5474, 1012, 1012,
        1012, 102,  0,    0,    0,     0,    0,     0,     0,    0,     0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,     0,     0,    0,     0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,     0,     0,    0,     0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,     0,     0,    0,     0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,     0,     0,    0,     0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,     0,     0,    0,     0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,     0,     0,    0,     0,    0,    0,    0,    0,    0,
    };
    const relevant_attention_mask = [_]i64{
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    // Time single inference
    const start = std.time.nanoTimestamp();
    const relevant_score = try session.infer(&relevant_input_ids, &relevant_attention_mask);
    const end = std.time.nanoTimestamp();

    const elapsed_ns = end - start;
    const elapsed_us = @divFloor(elapsed_ns, 1_000);
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_us)) / 1000.0;

    std.debug.print("\nRelevant pair score (pasta/pasta recipe): {d}\n", .{relevant_score});
    std.debug.print("Single inference time: {d}μs ({d:.2}ms)\n", .{ elapsed_us, elapsed_ms });

    // Second test pair: "how to make pasta" + "Machine learning is a subset of AI..."
    // This should score low (irrelevant match)
    const irrelevant_input_ids = [_]i64{
        101,  2129, 2000, 2191, 24857, 102,  3698, 4083, 2003, 1037, 16745, 1997, 9932, 1012, 1012, 1012,
        102,  0,    0,    0,    0,     0,    0,    0,    0,    0,    0,     0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,    0,    0,    0,    0,     0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,    0,    0,    0,    0,     0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,    0,    0,    0,    0,     0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,    0,    0,    0,    0,     0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,    0,    0,    0,    0,     0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,    0,    0,    0,    0,     0,    0,    0,    0,    0,
    };
    const irrelevant_attention_mask = [_]i64{
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    const irrelevant_score = try session.infer(&irrelevant_input_ids, &irrelevant_attention_mask);
    std.debug.print("Irrelevant pair score (pasta/ML): {d}\n", .{irrelevant_score});

    // Verify scores are valid and relevant pair scores higher than irrelevant
    try std.testing.expect(!std.math.isNan(relevant_score));
    try std.testing.expect(!std.math.isNan(irrelevant_score));
    try std.testing.expect(relevant_score > irrelevant_score);
}

test "Baseline throughput: 100 sequential inferences" {
    const allocator = std.testing.allocator;

    var session = try ONNXSession.init(allocator, "models/minilm.onnx");
    defer session.deinit();

    // Use the same test input for all inferences (realistic scenario)
    const input_ids = [_]i64{
        101,  2129, 2000, 2191, 24857, 102,  24857, 17974, 1024, 26077, 2300, 1010, 5587, 5474, 1012, 1012,
        1012, 102,  0,    0,    0,     0,    0,     0,     0,    0,     0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,     0,     0,    0,     0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,     0,     0,    0,     0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,     0,     0,    0,     0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,     0,     0,    0,     0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,     0,     0,    0,     0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,     0,     0,    0,     0,    0,    0,    0,    0,    0,
    };
    const attention_mask = [_]i64{
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    const num_inferences: usize = 100;
    const start = std.time.nanoTimestamp();

    var i: usize = 0;
    while (i < num_inferences) : (i += 1) {
        _ = try session.infer(&input_ids, &attention_mask);
    }

    const end = std.time.nanoTimestamp();
    const elapsed_ns = end - start;
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const per_inference_ms = elapsed_ms / @as(f64, @floatFromInt(num_inferences));
    const throughput = @as(f64, @floatFromInt(num_inferences)) / (elapsed_ms / 1000.0);

    std.debug.print("\n=== Baseline Throughput ===\n", .{});
    std.debug.print("100 sequential inferences: {d:.2}ms total\n", .{elapsed_ms});
    std.debug.print("Per-inference: {d:.2}ms\n", .{per_inference_ms});
    std.debug.print("Throughput: {d:.1} inferences/sec\n", .{throughput});
    std.debug.print("Projected time for 1000 docs: {d:.0}ms\n", .{per_inference_ms * 1000.0});
    std.debug.print("\nTarget: 50-100ms for 1000 docs (need {d:.0}x speedup)\n", .{(per_inference_ms * 1000.0) / 75.0});
}

test "Batched inference performance" {
    const allocator = std.testing.allocator;

    var session = try ONNXSession.init(allocator, "models/minilm.onnx");
    defer session.deinit();

    // batch_size=32 shows best per-inference performance on CPU
    // Larger batches (64+) hit memory bandwidth limits and get slower
    const batch_size: usize = 32;

    // Prepare test input (same as previous tests)
    const test_input_ids = [_]i64{
        101,  2129, 2000, 2191, 24857, 102,  24857, 17974, 1024, 26077, 2300, 1010, 5587, 5474, 1012, 1012,
        1012, 102,  0,    0,    0,     0,    0,     0,     0,    0,     0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,     0,     0,    0,     0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,     0,     0,    0,     0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,     0,     0,    0,     0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,     0,     0,    0,     0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,     0,     0,    0,     0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,     0,    0,     0,     0,    0,     0,    0,    0,    0,    0,    0,
    };
    const test_attention_mask = [_]i64{
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    // Build batch arrays
    var batch_input_ids_ptrs = try allocator.alloc([]const i64, batch_size);
    defer allocator.free(batch_input_ids_ptrs);
    var batch_attention_mask_ptrs = try allocator.alloc([]const i64, batch_size);
    defer allocator.free(batch_attention_mask_ptrs);

    for (0..batch_size) |i| {
        batch_input_ids_ptrs[i] = &test_input_ids;
        batch_attention_mask_ptrs[i] = &test_attention_mask;
    }

    // Warmup: run once to compile/optimize the model
    _ = try session.infer(&test_input_ids, &test_attention_mask);

    // Time batched inference
    const start_batch = std.time.nanoTimestamp();
    const batch_scores = try session.inferBatch(batch_input_ids_ptrs, batch_attention_mask_ptrs);
    const end_batch = std.time.nanoTimestamp();
    defer allocator.free(batch_scores);

    std.debug.print("\nFirst batch score: {d}\n", .{batch_scores[0]});

    const elapsed_batch_ns = end_batch - start_batch;
    const elapsed_batch_ms = @as(f64, @floatFromInt(elapsed_batch_ns)) / 1_000_000.0;
    const per_inference_batch_ms = elapsed_batch_ms / @as(f64, @floatFromInt(batch_size));

    // Compare to sequential inference time
    const start_seq = std.time.nanoTimestamp();
    for (0..batch_size) |i| {
        _ = try session.infer(batch_input_ids_ptrs[i], batch_attention_mask_ptrs[i]);
    }
    const end_seq = std.time.nanoTimestamp();

    const elapsed_seq_ns = end_seq - start_seq;
    const elapsed_seq_ms = @as(f64, @floatFromInt(elapsed_seq_ns)) / 1_000_000.0;
    const per_inference_seq_ms = elapsed_seq_ms / @as(f64, @floatFromInt(batch_size));

    const speedup = elapsed_seq_ms / elapsed_batch_ms;

    std.debug.print("\n=== Batch Inference Performance (batch_size={d}) ===\n", .{batch_size});
    std.debug.print("Batched:    {d:.2}ms total ({d:.3}ms per inference)\n", .{ elapsed_batch_ms, per_inference_batch_ms });
    std.debug.print("Sequential: {d:.2}ms total ({d:.3}ms per inference)\n", .{ elapsed_seq_ms, per_inference_seq_ms });
    std.debug.print("Speedup:    {d:.1}x faster\n", .{speedup});
    std.debug.print("\nProjected time for 1000 docs with batching: {d:.0}ms\n", .{per_inference_batch_ms * 1000.0});

    // Verify all scores are identical (same input)
    for (batch_scores) |score| {
        try std.testing.expect(!std.math.isNan(score));
        try std.testing.expectApproxEqRel(batch_scores[0], score, 0.001);
    }
}
