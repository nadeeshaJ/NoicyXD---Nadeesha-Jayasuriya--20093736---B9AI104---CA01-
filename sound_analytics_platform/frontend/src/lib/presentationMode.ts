const STORAGE_KEY = "sap-presentation-mode";

export function getPresentationMode(): boolean {
  return localStorage.getItem(STORAGE_KEY) === "1";
}

export function setPresentationMode(enabled: boolean): void {
  localStorage.setItem(STORAGE_KEY, enabled ? "1" : "0");
  applyPresentationMode(enabled);
}

export function applyPresentationMode(enabled: boolean): void {
  document.documentElement.classList.toggle("presentation-mode", enabled);
}
