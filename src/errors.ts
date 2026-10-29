/** Base class for all smart-docs errors. `hint` tells the agent how to recover. */
export class SmartDocsError extends Error {
  readonly hint: string;

  constructor(message: string, hint = "") {
    super(message);
    this.name = new.target.name;
    this.hint = hint;
  }

  /** One-line, agent-friendly recovery instructions. */
  toStringWithHint(): string {
    return this.hint ? `${this.message} — ${this.hint}` : this.message;
  }
}

export class InvalidPathError extends SmartDocsError {}
export class FileNotFoundError extends SmartDocsError {}
export class UnsupportedFileError extends SmartDocsError {}
export class BinaryFileError extends SmartDocsError {}
export class FileTooLargeError extends SmartDocsError {}
export class EmptyFileError extends SmartDocsError {}
export class EmptyIndexError extends SmartDocsError {}
export class EmptyDirError extends SmartDocsError {}
export class ModelMismatchError extends SmartDocsError {}
export class StoreCorruptError extends SmartDocsError {}
export class RateLimitedError extends SmartDocsError {}
