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
        exe.addIncludePath(.{ .cwd_relative = include_path });
    } else {
        // Default paths for different platforms
        if (std.process.hasEnvVarConstant("COLAB_GPU")) {
            // Google Colab with pip-installed onnxruntime-gpu
            exe.addIncludePath(.{ .cwd_relative = "/usr/local/lib/python3.10/dist-packages/onnxruntime/capi/include" });
        } else {
            // Mac Homebrew default
            exe.addIncludePath(.{ .cwd_relative = "/opt/homebrew/Cellar/onnxruntime/1.22.2_4/include" });
        }
    }

    if (onnx_lib) |lib_path| {
        defer b.allocator.free(lib_path);
        exe.addLibraryPath(.{ .cwd_relative = lib_path });
    } else {
        if (std.process.hasEnvVarConstant("COLAB_GPU")) {
            // Google Colab with pip-installed onnxruntime-gpu
            exe.addLibraryPath(.{ .cwd_relative = "/usr/local/lib/python3.10/dist-packages/onnxruntime/capi/lib" });
        } else {
            // Mac Homebrew default
            exe.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/Cellar/onnxruntime/1.22.2_4/lib" });
        }
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
    } else {
        if (std.process.hasEnvVarConstant("COLAB_GPU")) {
            unit_tests.addIncludePath(.{ .cwd_relative = "/usr/local/lib/python3.10/dist-packages/onnxruntime/capi/include" });
        } else {
            unit_tests.addIncludePath(.{ .cwd_relative = "/opt/homebrew/Cellar/onnxruntime/1.22.2_4/include" });
        }
    }

    if (test_onnx_lib) |lib_path| {
        defer b.allocator.free(lib_path);
        unit_tests.addLibraryPath(.{ .cwd_relative = lib_path });
    } else {
        if (std.process.hasEnvVarConstant("COLAB_GPU")) {
            unit_tests.addLibraryPath(.{ .cwd_relative = "/usr/local/lib/python3.10/dist-packages/onnxruntime/capi/lib" });
        } else {
            unit_tests.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/Cellar/onnxruntime/1.22.2_4/lib" });
        }
    }

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
