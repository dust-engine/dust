"""Rules for pumicite_cli: slang pipeline layout extraction and ron-to-bin conversion."""

def _slang_playout_impl(ctx):
    sdk = ctx.toolchains["@rules_vulkan//vulkan:toolchain_type"].info
    out = ctx.actions.declare_file(ctx.attr.out)

    args = ctx.actions.args()
    args.add("slang")
    args.add_all(ctx.files.srcs)
    args.add("-o", out)
    args.add("-f", ctx.attr.format)

    if ctx.attr.profile:
        args.add("-p", ctx.attr.profile)

    for set_attr in ctx.attr.set_attrs:
        args.add("--set-attr", set_attr)

    ctx.actions.run(
        inputs = ctx.files.srcs,
        outputs = [out],
        arguments = [args],
        executable = ctx.executable._tool,
        env = sdk.env,
        progress_message = "Extracting pipeline layout from %s" % ", ".join([f.short_path for f in ctx.files.srcs]),
        mnemonic = "SlangPlayout",
    )

    return [DefaultInfo(files = depset([out]))]

slang_playout = rule(
    implementation = _slang_playout_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = True,
            mandatory = True,
            doc = "Slang shader source files",
        ),
        "out": attr.string(
            mandatory = True,
            doc = "Output file name",
        ),
        "format": attr.string(
            default = "bin",
            doc = "Output format (ron, bin)",
        ),
        "profile": attr.string(
            doc = "Slang profile to compile against",
        ),
        "set_attrs": attr.string_list(
            doc = "Per-set DescriptorSetLayout attributes. " +
                  "Each entry is '<SET>:<ATTR>[,<ATTR>...]' where ATTR is one of: " +
                  "push_descriptor, update_after_bind_pool, descriptor_buffer. " +
                  'Example: ["0:push_descriptor", "1:update_after_bind_pool,descriptor_buffer"]',
        ),
        "_tool": attr.label(
            default = "@crates//:pumicite_cli__pumicite_cli",
            executable = True,
            cfg = "exec",
        ),
    },
    toolchains = ["@rules_vulkan//vulkan:toolchain_type"],
)

def _ron2bin_impl(ctx):
    sdk = ctx.toolchains["@rules_vulkan//vulkan:toolchain_type"].info
    out = ctx.actions.declare_file(ctx.attr.out)

    args = ctx.actions.args()
    args.add("ron2bin")
    args.add(ctx.file.src)
    args.add("-o", out)

    if ctx.attr.type:
        args.add("-t", ctx.attr.type)

    ctx.actions.run(
        inputs = [ctx.file.src],
        outputs = [out],
        arguments = [args],
        executable = ctx.executable._tool,
        env = sdk.env,
        progress_message = "Converting %s to bin" % ctx.file.src.short_path,
        mnemonic = "Ron2Bin",
    )

    return [DefaultInfo(files = depset([out]))]

ron2bin = rule(
    implementation = _ron2bin_impl,
    attrs = {
        "src": attr.label(
            allow_single_file = [".ron"],
            mandatory = True,
            doc = "Input .ron file",
        ),
        "out": attr.string(
            mandatory = True,
            doc = "Output file name",
        ),
        "type": attr.string(
            doc = "Type contained in the .ron file (inferred from extension if omitted). " +
                  "One of: pipeline-layout, descriptor-set-layout, compute-pipeline, " +
                  "graphics-pipeline, ray-tracing-pipeline",
        ),
        "_tool": attr.label(
            default = "@crates//:pumicite_cli__pumicite_cli",
            executable = True,
            cfg = "exec",
        ),
    },
    toolchains = ["@rules_vulkan//vulkan:toolchain_type"],
)
