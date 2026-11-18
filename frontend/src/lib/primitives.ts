import i18n from "../i18n";

// The training session API's primitives, and the ops that call them in the templates.
// What each one does is interface text (loop.prim in i18n/msg/loop.ts); "measured" figures
// there are from the 2026-10-27 live runs (research 261026, evidence/).
const OPS: { name: string; ops: string[] }[] = [
  { name: "create_lora_training_client", ops: ["create_session"] },
  { name: "sample", ops: ["sample"] },
  { name: "forward", ops: ["forward"] },
  { name: "forward_backward", ops: ["forward_backward"] },
  { name: "optim_step", ops: ["optim_step"] },
  { name: "compute_logprobs", ops: ["compute_logprobs", "teacher"] },
  { name: "save_weights_for_sampler", ops: ["save_weights_and_rebind", "save_weights_merged"] },
  { name: "save_state", ops: ["save_state"] },
  { name: "load_state", ops: ["load_state"] },
  { name: "stop", ops: ["stop_session"] },
];

/** The primitives with what each does, in the interface language. Call during render. */
export function primitives(): { name: string; ops: string[]; what: string }[] {
  const what = i18n.t("loop.prim", { returnObjects: true }) as Record<string, string>;
  return OPS.map((p) => ({ ...p, what: what[p.name] }));
}

/** Primitives a scenario's loop calls, from its loop description (nodes, and work before and after). */
export function primitivesOf(loop: { nodes: { op: string; primitive?: string }[]; before: string[]; after: string[] }): string[] {
  const ops = new Set([...loop.nodes.map((n) => n.op), ...loop.before, ...loop.after]);
  const named = new Set(loop.nodes.map((n) => n.primitive).filter(Boolean) as string[]);
  return OPS.filter((p) => named.has(p.name) || p.ops.some((o) => ops.has(o))).map((p) => p.name);
}
