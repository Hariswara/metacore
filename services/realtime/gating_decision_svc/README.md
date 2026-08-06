# gating_decision_svc

Serves Module 3's meta-policy in the sub-second reactive loop. Owner: Saabir S. (IT23432598).

Go has almost no ML ecosystem, so this service **never trains**. The policy is trained in
`services/learned/module3_metapolicy` and exported to ONNX; this service loads it and runs the hot
loop. That split is deliberate — see the stack-friction note in the scaffold review.

Start in Python and port here only if profiling shows the latency budget is missed. Do not build
Go services speculatively.
