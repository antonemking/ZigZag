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
    exe.addIncludePath(.{ .cwd_relative = "/opt/homebrew/Cellar/onnxruntime/1.22.2_4/include" });
    exe.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/Cellar/onnxruntime/1.22.2_4/lib" });

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

    // Link ONNX Runtime for tests too
    unit_tests.linkSystemLibrary("onnxruntime");
    unit_tests.linkLibC();
    unit_tests.addIncludePath(.{ .cwd_relative = "/opt/homebrew/Cellar/onnxruntime/1.22.2_4/include" });
    unit_tests.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/Cellar/onnxruntime/1.22.2_4/lib" });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
