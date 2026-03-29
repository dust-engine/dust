"""Rule to extract pipeline layout from a Slang shader using pumicite_slang."""

def _slang_playout_impl(ctx):
    sdk = ctx.toolchains["@rules_vulkan//vulkan:toolchain_type"].info
    out = ctx.actions.declare_file(ctx.attr.out)

    args = ctx.actions.args()
    args.add_all(ctx.files.srcs)
    args.add("-o", out)
    args.add("-f", ctx.attr.format)

    if ctx.attr.profile:
        args.add("-p", ctx.attr.profile)

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
            default = "postcard",
            doc = "Output format (ron, postcard)",
        ),
        "profile": attr.string(
            doc = "Slang profile to compile against",
        ),
        "_tool": attr.label(
            default = "@crates//:pumicite_slang__pumicite_slang",
            executable = True,
            cfg = "exec",
        ),
    },
    toolchains = ["@rules_vulkan//vulkan:toolchain_type"],
)
