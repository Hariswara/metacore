// Command gating serves Module 3's exported policy in the hot loop.
//
// This service never trains. Training happens in services/learned/module3_metapolicy; the policy
// is exported to ONNX and loaded here so the latency-critical path stays in Go.
package main

import "log"

func main() {
	log.Println("gating_decision_svc: scaffold — no policy loaded yet")
}
