/** The same shape as T, with every leaf a string: a translation must have exactly T's keys. */
export type Strings<T> = { [K in keyof T]: T[K] extends string ? string : Strings<T[K]> };

/**
 * One namespace of interface text, English and Chinese side by side. English is the source;
 * the Chinese object is checked against it, so a missing or extra key fails `tsc`.
 */
export const bilingual = <T extends object>(en: T, zh: Strings<T>) => ({ en, zh });
