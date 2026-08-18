// Flat config. ESLint was invoked by the lint script but never declared as a dependency, so it
// resolved to whatever happened to be installed globally — here, a 2019 build that cannot read
// TypeScript at all. Declared and configured now, matching how task and buf are handled at the
// repo root.
import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist/**"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["src/**/*.{ts,tsx}"],
    plugins: { "react-hooks": reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
      // Charts index into fixed-length arrays the type system cannot narrow; the alternative is
      // non-null assertions scattered through the render path, which is worse.
      "@typescript-eslint/no-non-null-assertion": "off",
    },
  },
);
