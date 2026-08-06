# Base images

Shared base layers so per-service Dockerfiles stay short and a build does not reinstall the same
toolchain four times. Keep the `deterministic-core` base separate from the learned base — it must
not inherit an image that carries ML libraries.
