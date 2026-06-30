const EQUIVALENT_LABELS = new Set(["dog_bark|dog", "dog|dog_bark"]);

export function labelsMatch(predicted: string | null | undefined, groundTruth: string | null | undefined): boolean {
  if (!predicted || !groundTruth) return false;
  if (predicted === groundTruth) return true;
  return EQUIVALENT_LABELS.has(`${predicted}|${groundTruth}`);
}

export function hasGroundTruth(row: {
  has_ground_truth?: boolean;
  ground_truth_label?: string | null;
}): boolean {
  return Boolean(row.has_ground_truth && row.ground_truth_label);
}
