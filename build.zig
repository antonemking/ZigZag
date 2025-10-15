const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "zigzag",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Link ONNX Runtime
    exe.linkSystemLibrary("onnxruntime");
    exe.linkLibC();

    // Detect platform and set ONNX Runtime paths
    // User can override with ONNX_INCLUDE and ONNX_LIB environment variables
    const onnx_include = std.process.getEnvVarOwned(b.allocator, "ONNX_INCLUDE") catch null;
    const onnx_lib = std.process.getEnvVarOwned(b.allocator, "ONNX_LIB") catch null;

    if (onnx_include) |include_path| {
        defer b.allocator.free(include_path);
        // Include both base path and onnxruntime subdirectory
        exe.addIncludePath(.{ .cwd_relative = include_path });
        // Try adding /onnxruntime subdir if it exists (for downloaded headers)
        const subdir = std.fmt.allocPrint(b.allocator, "{s}/onnxruntime", .{include_path}) catch include_path;
        defer if (subdir.ptr != include_path.ptr) b.allocator.free(subdir);
        exe.addIncludePath(.{ .cwd_relative = subdir });
    } else {
        // Default: Mac Homebrew
        exe.addIncludePath(.{ .cwd_relative = "/opt/homebrew/Cellar/onnxruntime/1.22.2_4/include" });
    }

    if (onnx_lib) |lib_path| {
        defer b.allocator.free(lib_path);
        exe.addLibraryPath(.{ .cwd_relative = lib_path });
    } else {
        // Default: Mac Homebrew
        exe.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/Cellar/onnxruntime/1.22.2_4/lib" });
    }

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Tests
    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Link ONNX Runtime for tests too (same paths as exe)
    unit_tests.linkSystemLibrary("onnxruntime");
    unit_tests.linkLibC();

    const test_onnx_include = std.process.getEnvVarOwned(b.allocator, "ONNX_INCLUDE") catch null;
    const test_onnx_lib = std.process.getEnvVarOwned(b.allocator, "ONNX_LIB") catch null;

    if (test_onnx_include) |include_path| {
        defer b.allocator.free(include_path);
        unit_tests.addIncludePath(.{ .cwd_relative = include_path });
        const subdir = std.fmt.allocPrint(b.allocator, "{s}/onnxruntime", .{include_path}) catch include_path;
        defer if (subdir.ptr != include_path.ptr) b.allocator.free(subdir);
        unit_tests.addIncludePath(.{ .cwd_relative = subdir });
    } else {
        // Default: Mac Homebrew
        unit_tests.addIncludePath(.{ .cwd_relative = "/opt/homebrew/Cellar/onnxruntime/1.22.2_4/include" });
    }

    if (test_onnx_lib) |lib_path| {
        defer b.allocator.free(lib_path);
        unit_tests.addLibraryPath(.{ .cwd_relative = lib_path });
    } else {
        // Default: Mac Homebrew
        unit_tests.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/Cellar/onnxruntime/1.22.2_4/lib" });
    }

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
