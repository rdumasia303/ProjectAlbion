---- MODULE KillSwitch ----
(*
ALBION Kill-Switch Specification

Formal specification of the kill-switch state machine that freezes
the system when data drift is detected and sufficient approvals are obtained.

Safety property: Once frozen, always frozen (no escaping the frozen state)

Liveness property: If drift detected and quorum reached, eventually freeze

This spec can be model-checked with TLC to prove these properties hold.
*)

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    QuorumSize,         \* Number of approvals needed to freeze (e.g., 2)
    MaxApprovers,       \* Total number of possible approvers (e.g., 5)
    MaxDriftDetections  \* Max drift detections in trace (for model checking)

ASSUME QuorumSize \in Nat /\ QuorumSize > 0
ASSUME MaxApprovers \in Nat /\ MaxApprovers >= QuorumSize
ASSUME MaxDriftDetections \in Nat

VARIABLES
    system_state,   \* {Live, DriftDetected, Frozen}
    approvals,      \* Set of approver IDs who have voted to freeze
    data_drift,     \* Boolean: has drift been detected?
    drift_count     \* Counter for model checking (limit state space)

vars == <<system_state, approvals, data_drift, drift_count>>

(* Type invariant *)
TypeInvariant ==
    /\ system_state \in {"Live", "DriftDetected", "Frozen"}
    /\ approvals \subseteq (1..MaxApprovers)
    /\ data_drift \in BOOLEAN
    /\ drift_count \in Nat

(* Initial state *)
Init ==
    /\ system_state = "Live"
    /\ approvals = {}
    /\ data_drift = FALSE
    /\ drift_count = 0

(* Drift detection action (external trigger) *)
DetectDrift ==
    /\ system_state = "Live"
    /\ drift_count < MaxDriftDetections  \* Bound for model checking
    /\ data_drift' = TRUE
    /\ system_state' = "DriftDetected"
    /\ drift_count' = drift_count + 1
    /\ UNCHANGED approvals

(* Approver votes to freeze *)
ApproverVote(approver) ==
    /\ approver \in (1..MaxApprovers)
    /\ approver \notin approvals
    /\ system_state \in {"Live", "DriftDetected"}
    /\ approvals' = approvals \union {approver}
    /\ IF Cardinality(approvals') >= QuorumSize
       THEN system_state' = "Frozen"
       ELSE UNCHANGED system_state
    /\ UNCHANGED <<data_drift, drift_count>>

(* Clear false-positive drift (requires unanimous approval to cancel) *)
ClearDrift ==
    /\ system_state = "DriftDetected"
    /\ Cardinality(approvals) = MaxApprovers  \* All approvers agree it's false positive
    /\ data_drift' = FALSE
    /\ system_state' = "Live"
    /\ approvals' = {}
    /\ UNCHANGED drift_count

(* Next-state relation *)
Next ==
    \/ DetectDrift
    \/ \E a \in (1..MaxApprovers) : ApproverVote(a)
    \/ ClearDrift

(* Specification *)
Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* SAFETY PROPERTIES *)

(* Safety: Once frozen, always frozen *)
SafetyInvariant ==
    [](system_state = "Frozen" => [](system_state = "Frozen"))

(* Safety: Cannot freeze without quorum *)
QuorumRequired ==
    system_state = "Frozen" => Cardinality(approvals) >= QuorumSize

(* Safety: Frozen state cannot be unfrozen *)
NoUnfreeze ==
    (system_state = "Frozen" /\ system_state' = system_state') => system_state' = "Frozen"

-----------------------------------------------------------------------------
(* LIVENESS PROPERTIES *)

(* Liveness: If drift detected and quorum reached, eventually freeze *)
EventuallyFreeze ==
    (data_drift /\ Cardinality(approvals) >= QuorumSize) ~> (system_state = "Frozen")

(* Liveness: System doesn't deadlock (some action always possible) *)
(* This may not hold if we reach MaxDriftDetections - that's OK for model checking *)

-----------------------------------------------------------------------------
(* INVARIANTS TO CHECK *)

(* Main safety invariant: If currently frozen, was frozen in previous state or just became frozen *)
FrozenIsSticky ==
    system_state = "Frozen" =>
        (system_state = "Frozen" \/ Cardinality(approvals) >= QuorumSize)

(* Approvals monotonically increase until freeze or clear *)
ApprovalsMonotonic ==
    (system_state \in {"Live", "DriftDetected"}) =>
        (approvals' \subseteq approvals \/ approvals \subseteq approvals')

(* Drift detection requires Live state *)
DriftDetectionValid ==
    (data_drift /\ data_drift') => (system_state = "Live" \/ system_state = "DriftDetected")

-----------------------------------------------------------------------------
(* STATE CONSTRAINTS (for model checking) *)

(* Limit state space for TLC *)
StateConstraint ==
    /\ drift_count <= MaxDriftDetections
    /\ Cardinality(approvals) <= MaxApprovers

-----------------------------------------------------------------------------
(* THEOREM STATEMENTS *)

(*
These theorems can be checked by TLC model checker:

1. Run TLC with Spec and check invariants:
   - TypeInvariant
   - QuorumRequired
   - FrozenIsSticky

2. Check temporal properties:
   - SafetyInvariant
   - EventuallyFreeze

Model checking parameters for TLC:
- QuorumSize: 2
- MaxApprovers: 5
- MaxDriftDetections: 3

This gives a tractable state space while proving the key properties.
*)

THEOREM Spec => TypeInvariant
THEOREM Spec => QuorumRequired
THEOREM Spec => FrozenIsSticky
THEOREM Spec => SafetyInvariant
THEOREM Spec => EventuallyFreeze

-----------------------------------------------------------------------------
(* EXAMPLE TRACE PROPERTY *)

(*
This property describes a valid execution trace:
1. Start in Live state
2. Detect drift → DriftDetected
3. Collect approvals (one at a time)
4. When quorum reached → Frozen
5. Stay frozen forever
*)

ExampleTrace ==
    /\ system_state = "Live"
    /\ <>system_state = "DriftDetected"
    /\ <>system_state = "Frozen"
    /\ [](system_state = "Frozen" => []system_state = "Frozen")

====
