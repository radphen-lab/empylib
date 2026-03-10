from typing import Optional, List, Dict, Any, Iterable, Tuple
import csv, re
import pandas as pd
from pathlib import Path
import re

def _sniff_dialect(lines: List[str]):
    """
    Try to detect the delimiter style (dialect) used in a text/CSV-like file.

    Parameters
    ----------
    lines : List[str]
        List of lines from the file (usually read in advance).
        Only the first few lines are used for detection.

    Returns
    -------
    csv.Dialect
        A `Dialect` object describing how the file is structured
        (delimiter, quotechar, etc.), suitable for passing to `csv.reader`.

    How it works
    ------------
    - Joins up to the first 10 lines of the file into a single string.
    - Uses Python's built-in `csv.Sniffer.sniff` to guess whether the file
      uses commas, tabs, or semicolons as delimiters.
    - If detection fails (raises `csv.Error`), falls back to a simple custom
      dialect that assumes comma-delimited text with standard quoting rules.

    Limitations
    -----------
    - Only the first 10 lines are considered. If the real tabular data appears
      later in the file (e.g. PerkinElmer `.asc` files with long headers),
      the sniffer may fail or guess incorrectly.
    - Later parts of the parsing pipeline (_clean_rows) add extra safeguards,
      such as whitespace-splitting rows that didn’t parse properly.

    Notes
    -----
    For maximum robustness, you could extend this function to look beyond the
    first 10 lines (or specifically locate numeric/tabular rows) before calling
    `csv.Sniffer`. This current implementation is a "good enough" heuristic
    for most Shimadzu/PerkinElmer exports.
    """
    # Take at most the first 10 lines of the file as a detection sample
    sample = "\n".join(lines[:10])

    try:
        # Let the built-in Sniffer guess the delimiter
        return csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        # If detection fails, define a minimal fallback dialect
        class _D(csv.Dialect):
            delimiter = ","        # assume comma-separated
            quotechar = '"'        # allow quoted strings with "
            escapechar = None
            doublequote = True
            skipinitialspace = False
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL
        return _D()

def _tokenize_lines(lines: List[str]) -> List[List[str]]:
    """
    Convert a list of raw text lines into tokenized (split) rows.

    Parameters
    ----------
    lines : List[str]
        Raw text lines from a UV-Vis data file (Shimadzu, PerkinElmer, etc.).

    Returns
    -------
    List[List[str]]
        A list of rows, where each row is itself a list of string tokens
        representing the fields (columns) in that line.

    How it works
    ------------
    1. Uses `_sniff_dialect` to guess the delimiter (comma, tab, semicolon).
    2. Passes all lines through Python's `csv.reader` with that dialect.
    3. For each row, strips whitespace and surrounding quotes from fields.
    4. Drops empty strings (which can occur if there are extra delimiters).
    5. If the row only has one field but that field still contains spaces/tabs,
       falls back to splitting on whitespace with a regex.
       (This handles cases like PerkinElmer `.asc` files that are space-aligned.)
    6. Returns a list of cleaned rows.

    Notes
    -----
    - This function is deliberately forgiving: it tries to return
      "something usable" regardless of delimiter style.
    - You may still need to post-process the tokens into floats later.
    """
    # Step 1: try to guess the delimiter using first 10 lines
    dialect = _sniff_dialect(lines)

    rows: List[List[str]] = []
    # Step 2: feed all lines into the CSV reader
    for row in csv.reader(lines, dialect):
        # Step 3: clean up each field by stripping spaces and quotes
        cleaned = [field.strip().strip('"').strip("'") for field in row]
        # Step 4: remove empty fields (from multiple delimiters, etc.)
        cleaned = [c for c in cleaned if c != ""]

        # Step 5: fallback — if we only got one field but it still has spaces/tabs,
        # split it again using whitespace regex.
        if len(cleaned) == 1 and (" " in cleaned[0] or "\t" in cleaned[0]):
            toks = re.split(r"\s+", cleaned[0].strip())
            cleaned = [t for t in toks if t]  # drop empty tokens

        # Collect the cleaned row
        rows.append(cleaned)

    return rows

def _is_two_numeric(row: List[str]) -> bool:
    """
    Check whether a row contains at least two numeric-like values.

    Parameters
    ----------
    row : List[str]
        A tokenized row (list of strings), typically produced by `_tokenize_lines`.

    Returns
    -------
    bool
        True if the row has at least two elements and the first two can be
        safely converted to floats (after removing percent signs and trimming
        whitespace). False otherwise.

    Notes
    -----
    - This is used to locate the start of the numeric data region in a file
      (e.g. skipping headers until the first row of actual tabulated numbers).
    - It tolerates values like "220.0", "0.164", or "85.5%".
    - Any exception during float conversion results in a False return.
    """
    # Must have at least two tokens to test
    if len(row) < 2:
        return False
    try:
        # Try to parse the first two entries as floats
        float(row[0].replace("%", "").strip())
        float(row[1].replace("%", "").strip())
        return True
    except Exception:
        # If conversion fails (non-numeric strings, missing values, etc.)
        return False

from typing import List, Optional

def _first_numeric_idx(rows: List[List[str]]) -> Optional[int]:
    """
    Find the index of the first row that looks like numeric data.

    Parameters
    ----------
    rows : List[List[str]]
        A list of tokenized rows (each row is a list of strings),
        typically produced by `_tokenize_lines`.

    Returns
    -------
    Optional[int]
        The index (0-based) of the first row where `_is_two_numeric(row)` is True,
        meaning the row appears to contain at least two numeric-like values.
        Returns None if no such row is found.

    Notes
    -----
    - This function is useful for skipping over header / metadata lines
      until the start of the tabulated UV-Vis data.
    - It relies on `_is_two_numeric` to decide whether a row qualifies.
    - If no numeric row is found, the caller should handle the None case.
    
    Examples
    --------
    >>> _first_numeric_idx([["Sample", "Name"],
    ...                     ["220.0", "0.164"],
    ...                     ["221.0", "0.155"]])
    1
    """
    # Walk through each row, keeping track of its index
    for i, r in enumerate(rows):
        # As soon as a row looks numeric, return its index
        if _is_two_numeric(r):
            return i
    # If no numeric rows found, return None
    return None

def _to_float(tok: str) -> float:
    """
    Convert a string token to float, ignoring percent signs and whitespace.

    Parameters
    ----------
    tok : str
        Input string, e.g. "85.2%", "  220.0 ".

    Returns
    -------
    float
        Numeric value parsed from the string.
    """
    return float(tok.replace("%", "").strip())

def _read_shimadzu_raw(path: str) -> Dict[str, Any]:
    """
    Read a Shimadzu UV-Vis exported text file and extract raw components.

    This function does NOT build a DataFrame. It only returns the pieces
    needed for further processing (headers, metadata, and tokenized data).

    Parameters
    ----------
    path : str
        Path to a Shimadzu `.txt` file.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - "instrument": str, always "Shimadzu"
        - "sample_name": str or None, sample name from the first header line
        - "header_lines": List[str], raw header lines (after removing comments)
        - "header_rows": List[List[str]], tokenized header rows
        - "data_rows": List[List[str]], tokenized data rows (tabulated values)
        - "col1_from_file": str or None, first column name from the file
        - "col2_from_file": str or None, second column name from the file

    Notes
    -----
    - Lines starting with '#' are treated as comments and discarded.
    - The "sample name" is assumed to be the first single-field header row.
    - Column headers are taken from the last header row with >= 2 fields.
    - Actual column naming/normalization and DataFrame construction are
      handled later by `read_uvvis`.
    """
    # Read all non-empty lines from the file, stripping whitespace
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        all_lines = [ln.strip() for ln in f if ln.strip()]

    # Remove comment lines (those starting with '#')
    non_comment = [ln for ln in all_lines if not ln.lstrip().startswith("#")]

    # Tokenize remaining lines into columns
    rows = _tokenize_lines(non_comment)

    # Locate the first numeric row (start of data block)
    first_data_idx = _first_numeric_idx(rows)
    if first_data_idx is None:
        raise ValueError("Shimadzu: could not locate numeric data.")

    # Split into header rows (before numeric data) and data rows (numeric table)
    header_rows = rows[:first_data_idx]
    data_rows  = rows[first_data_idx:]

    # Initialize metadata
    sample_name = None
    col1_from_file = None
    col2_from_file = None

    # Detect sample name and header column names
    if header_rows:
        # If the very first header row is a single field, treat it as sample name
        if len(header_rows[0]) == 1:
            sample_name = header_rows[0][0]
        # If any header row has 2+ fields, assume it contains column names
        hdr2p = [r for r in header_rows if len(r) >= 2]
        if hdr2p:
            hdr = hdr2p[-1]  # take the last one, closest to the data block
            col1_from_file = hdr[0].strip() or None
            col2_from_file = hdr[1].strip() or None

    # Package raw components into a dictionary
    return {
        "instrument": "Shimadzu",
        "sample_name": sample_name,
        "header_lines": non_comment[:first_data_idx],  # keep original header as strings
        "header_rows": header_rows,
        "data_rows": data_rows,
        "col1_from_file": col1_from_file,
        "col2_from_file": col2_from_file,
    }


def _read_perkinelmer_raw(path: str) -> Dict[str, Any]:
    """
    Read a PerkinElmer UV-Vis exported ASCII (.asc) file and extract raw components.

    This function does NOT build a DataFrame. It only extracts and returns
    the pieces of information needed for later processing, leaving column
    naming, normalization, and DataFrame construction to `read_uvvis`.

    Parameters
    ----------
    path : str
        Path to a PerkinElmer `.asc` file.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "instrument": str
            Always "PerkinElmer".
        - "sample_name": str or None
            Sample name if detected from header lines (keywords like "Sample", "Title", "Name").
        - "header_lines": List[str]
            Raw header lines before the numeric block.
        - "header_rows": List[List[str]]
            Tokenized header lines (split into fields).
        - "data_rows": List[List[str]]
            Tokenized numeric data rows (tabulated values).
        - "col1_from_file": str or None
            First column name if detected in header.
        - "col2_from_file": str or None
            Second column name if detected in header.

    Notes
    -----
    - PerkinElmer `.asc` files often have long headers (dozens of lines) before
      the numeric data starts. These headers may include instrument info,
      operator, date, etc.
    - Numeric data is usually whitespace-delimited, but may also include
      commas or semicolons depending on export options.
    - This function attempts to locate the first numeric row by checking
      for two float-like tokens. A regex fallback is used if the initial
      detection fails.
    - Column names are guessed from the last header row with >= 2 tokens,
      but may be missing or ambiguous.
    """
    # Read all non-empty lines from the file
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        all_lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    # Tokenize lines using CSV sniffer + fallback to whitespace splitting
    rows = _tokenize_lines(all_lines)

    # Try to find where numeric data starts
    first_data_idx = _first_numeric_idx(rows)
    if first_data_idx is None:
        # Fallback: regex match for a line with at least two numbers
        num_re = re.compile(
            r"^\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)"
            r"\s+([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)(\s|,|;|$)"
        )
        for i, ln in enumerate(all_lines):
            if num_re.match(ln):
                first_data_idx = i
                break
    if first_data_idx is None:
        raise ValueError("PerkinElmer: could not locate numeric data.")

    # Separate header and data portions
    header_lines = all_lines[:first_data_idx]        # raw strings
    header_rows  = _tokenize_lines(header_lines)     # tokenized header lines
    data_rows    = _tokenize_lines(all_lines[first_data_idx:])  # tokenized numeric block

    # Try to extract sample name from header (keywords: Sample, Title, Name)
    sample_name = None
    for ln in header_lines:
        m = re.search(r"(?:Sample(?:\s*ID)?|Title|Name)\s*[:=]\s*(.+)", ln, re.IGNORECASE)
        if m:
            sample_name = m.group(1).strip().strip('"')
            break

    # Try to extract column names from last header row with ≥ 2 fields
    col1_from_file, col2_from_file = None, None
    hdr2p = [r for r in header_rows if len(r) >= 2]
    if hdr2p:
        hdr = hdr2p[-1]
        col1_from_file = hdr[0].strip() or None
        col2_from_file = hdr[1].strip() or None

    # Return raw pieces (no DataFrame yet)
    return {
        "instrument": "PerkinElmer",
        "sample_name": sample_name,
        "header_lines": header_lines,
        "header_rows": header_rows,
        "data_rows": data_rows,
        "col1_from_file": col1_from_file,
        "col2_from_file": col2_from_file,
    }

# ---- Derive missing components per family: tot = dif + spec ----
def _derive_family(df: pd.DataFrame, tot: str, dif: str, spec: str) -> None:
    have = {c for c in (tot, dif, spec) if c in df.columns}
    if len(have) == 2:
        if tot not in have:
            df[tot] = df[spec] + df[dif]
        elif dif not in have:
            df[dif] = df[tot] - df[spec]
        elif spec not in have:
            df[spec] = df[tot] - df[dif]
    # 3 present → nothing to do; 1 present → insufficient to compute

    return df

def read_uvvis(path: str,
               vendor: Optional[str] = None,
               col1_name: Optional[str] = None,
               col2_name: Optional[str] = None) -> pd.DataFrame:
    """
    Unified UV-Vis reader for Shimadzu (.txt) and PerkinElmer (.asc) files.

    This is the public entry point that dispatches to vendor-specific
    raw readers (hidden functions), then applies common logic to build
    a tidy `pandas.DataFrame`.

    Parameters
    ----------
    path : str
        Path to the UV-Vis data file.
    
    vendor : str, optional
        Explicit vendor name ("shimadzu" or "perkinelmer").
        If not given, it is inferred from the file extension.
    
    col1_name : str, optional
        Override for the first column name (x-axis).
        Default is "wavelength (µm)" (converted from nm).
    
    col2_name : str, optional
        Override for the second column name (y-axis).
        Default is taken from the file if available, otherwise inferred.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by wavelength in micrometers (µm),
        with one measurement column scaled to 0–1.
        Metadata is stored in `df.attrs`:
        - "instrument": vendor name
        - "sample_name": sample name if found
        - "header_lines": raw header lines
    """
    # --- Vendor dispatch ---
    p = Path(path)
    v = (vendor or "").strip().lower()

    if not v:
        if p.suffix.lower() == ".txt":
            v = "shimadzu"
        elif p.suffix.lower() == ".asc":
            v = "perkinelmer"
        else:
            raise ValueError(f"Unknown extension '{p.suffix}'. "
                             "Use vendor='shimadzu' or 'perkinelmer'.")

    if v in ("shimadzu", "shimadzy", "shimazu"):
        raw = _read_shimadzu_raw(str(p))
    elif v in ("perkinelmer", "perkin", "perkin-elmer"):
        raw = _read_perkinelmer_raw(str(p))
    else:
        raise ValueError(f"Unsupported vendor '{vendor}'.")

    # --- Column naming ---
    # Wavelength column name (now µm by default)
    wavename = col1_name.strip() if isinstance(col1_name, str) and col1_name.strip() else "wavelength (µm)"

    # Second column → user override > vendor-provided > heuristics
    second = col2_name.strip() if isinstance(col2_name, str) and col2_name.strip() else raw.get("col2_from_file")

    if second is None:
        hdr_text = " ".join(raw.get("header_lines", []))
        if re.search(r"%\s*T|transmittance", hdr_text, re.IGNORECASE):
            second = "T%"
        elif re.search(r"reflect|R[:%]?", hdr_text, re.IGNORECASE):
            second = "R%"
        elif re.search(r"\babs\b|absorb", hdr_text, re.IGNORECASE):
            second = "Absorbance"
        else:
            if raw.get("instrument") == "Shimadzu":
                raise ValueError("Shimadzu: second column name not found. Pass col2_name.")
            second = "value"

    # --- Parse numeric data ---
    wl_vals: List[float] = []
    y_vals: List[float] = []
    for row in raw["data_rows"]:
        if len(row) >= 2:
            try:
                wl_vals.append(_to_float(row[0]))   # nm
                y_vals.append(_to_float(row[1]))   # raw % or value
            except Exception:
                continue
    if not wl_vals:
        raise ValueError(f"{raw['instrument']}: no numeric data rows parsed.")

    # --- Unit scaling ---
    # Convert wavelength nm → µm
    wl_vals = [w / 1000.0 for w in wl_vals]

    # Convert measurement to 0–1 if it looks like a percentage
    # Criteria: header contains '%' OR max value > 1
    if "%" in second or max(y_vals) > 1.0:
        y_vals = [y / 100.0 for y in y_vals]
        # Remove '%' from column name if present
        second = second.replace("%", "").strip()
        if not second:
            second = "value (fraction)"

    # --- Build DataFrame ---
    df = pd.DataFrame({wavename: wl_vals, second: y_vals}).set_index(wavename)

    # Attach metadata
    df.attrs["instrument"]   = raw.get("instrument")
    df.attrs["sample_name"]  = raw.get("sample_name")
    # df.attrs["header_lines"] = raw.get("header_lines")

    return df

def find_uvvis_samples(
    search_dirs: Optional[Iterable[str]] = None,
    tags: Optional[List[str]] = None,
    aliases: Optional[Dict[str, List[str]]] = None,
    exts: Tuple[str, ...] = (".txt", ".asc"),
) -> List[str]:
    """
    Find UV-Vis sample names by scanning files named <tag>_<sample><ext>.
    - Case-insensitive for tags/aliases and extensions.
    - Keeps the sample name exactly as it appears in filenames (after the first separator).
    
    Parameters
    ----------
    search_dirs : Iterable[str], optional
        List of directories to search for files.
        If None, defaults to ["."], the current directory.
    
    tags : List[str], optional
        List of tags to look for in filenames.
        If None, defaults to:
        ["Rtot", "Ttot", "Rspec", "Tspec", "Rdif", "Tdif"].
    
    aliases : Dict[str, List[str]], optional
        Mapping of tag → list of alternative names to try.
        If None, defaults to:
        {
            "Rtot": ["Rtot"],
            "Ttot": ["Ttot"],
            "Rspec": ["Rspec", "R_spec", "Rspecular"],
            "Tspec": ["Tspec", "T_spec", "Tspecular"],
            "Rdif": ["Rdif", "Rdiff", "Rdiffuse"],
            "Tdif": ["Tdif", "Tdiff", "Tdiffuse"],
        }.
    exts : Tuple[str, ...], optional
        Tuple of file extensions to consider (case-insensitive).
        Defaults to (".txt", ".asc").
    Returns
    -------
    List[str]
        List of unique sample names found (order not guaranteed).
    """
    if search_dirs is None:
        search_dirs = ["."]
    if tags is None:
        tags = ["Rtot", "Ttot", "Rspec", "Tspec", "Rdif", "Tdif"]
    if aliases is None:
        aliases = {
            "Rtot": ["Rtot"],
            "Ttot": ["Ttot"],
            "Rspec": ["Rspec", "R_spec", "Rspecular"],
            "Tspec": ["Tspec", "T_spec", "Tspecular"],
            "Rdif": ["Rdif", "Rdiff", "Rdiffuse"],
            "Tdif": ["Tdif", "Tdiff", "Tdiffuse"],
        }

    # Build a prefix-regex from all aliases (start of string, tag, then separators)
    alias_flat = set()
    for t in tags:
        for a in aliases.get(t, [t]):
            alias_flat.add(a)
    # sort by length desc to avoid partial matches (e.g., 'T' before 'Ttot')
    alias_sorted = sorted(alias_flat, key=len, reverse=True)
    tag_re = re.compile(rf"^({'|'.join(re.escape(a) for a in alias_sorted)})[ _-]+", re.IGNORECASE)

    samples: List[str] = []
    for d in search_dirs:
        base = Path(d)
        if not base.exists():
            continue
        for p in base.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in {e.lower() for e in exts}:
                continue
            stem = p.stem
            # remove '<tag><sep>' at start; keep the rest verbatim as the sample name
            sample_name = tag_re.sub("", stem)
            if sample_name and sample_name not in samples:
                samples.append(sample_name)
    return samples

def sample_uvvis(sample: str = None,
                 search_dirs: Optional[Iterable[str]] = None,
                 tags: Optional[List[str]] = None,
                 aliases: Optional[Dict[str, List[str]]] = None,
                 exts: Tuple[str, ...] = (".txt", ".asc"),
                 ref_standard: str = "spectralon") -> pd.DataFrame:
    """
    Aggregate UV-Vis measurements for a given sample into a single DataFrame.
    Applies reflectance correction using a standard reference spectrum


    Also derives the third component per family (R*, T*) via: total = diffuse + specular.

    Files searched (case-insensitive), by default:
      Rtot, Ttot, Rspec, Tspec, Rdif, Tdif
    
    Exact-name patterns per tag (sample kept exactly as passed):
      <tag>_<sample>.txt | <tag>_<sample>.asc
      <tag_lower>_<sample>.txt | <tag_lower>_<sample>.asc
      
    Parameters
    ----------
    sample : str, optional
        Sample name to look for in filenames (e.g. "MySample").
        If None, defaults to the sample files stored in "search_dirs".
    
    search_dirs : Iterable[str], optional
        List of directories to search for files.
        If None, defaults to ["."], the current directory.
    
    tags : List[str], optional
        List of tags to look for in filenames.
        If None, defaults to:
        ["Rtot", "Ttot", "Rspec", "Tspec", "Rdif", "Tdif"].
    
    aliases : Dict[str, List[str]], optional
        Mapping of tag → list of alternative names to try.
        If None, defaults to:
        {
            "Rtot": ["Rtot"],
            "Ttot": ["Ttot"],
            "Rspec": ["Rspec", "R_spec", "Rspecular"],
            "Tspec": ["Tspec", "T_spec", "Tspecular"],
            "Rdif": ["Rdif", "Rdiff", "Rdiffuse"],
            "Tdif": ["Tdif", "Tdiff", "Tdiffuse"],
        }.
    exts : Tuple[str, ...], optional
        Tuple of file extensions to consider (case-insensitive).
        Defaults to (".txt", ".asc").

    ref_standard : str, optional
        Path to the reference standard spectrum CSV file.
        Default is "spectralon.csv"

    Returns
    -------
    pandas.DataFrame
        Indexed by wavelength (µm). Columns among:
        Rtot, Ttot, Rspec, Tspec, Rdif, Tdif.
        Values are fractions (0–1) for percent quantities (handled in `read_uvvis`).

    Notes
    -----
    - If wavelength arrays differ across files, the function interpolates each
      column onto the union of all wavelengths (interior-only; no extrapolation).
    
    - If exactly two of (tot, dif, spec) are present in a family, the third is
      computed using tot = dif + spec. If all three exist, nothing is computed.
    """
    # Default to finding all samples if none specified
    if tags is None:
        tags = ["Rtot", "Ttot", "Rspec", "Tspec", "Rdif", "Tdif"]
    if search_dirs is None:
        search_dirs = ["."]
    if aliases is None:
        aliases = {
            "Rtot": ["Rtot"],
            "Ttot": ["Ttot"],
            "Rspec": ["Rspec", "R_spec", "Rspecular"],
            "Tspec": ["Tspec", "T_spec", "Tspecular"],
            "Rdif": ["Rdif", "Rdiff", "Rdiffuse"],
            "Tdif": ["Tdif", "Tdiff", "Tdiffuse"],
        }

    # ---- Find first matching file per tag (exact filename; sample kept exact) ----
    found: Dict[str, Path] = {}
    for tag in tags:
        for alias in aliases.get(tag, [tag]):
            found_in_any_dir = False
            for d in search_dirs:
                base = Path(d)
                candidates = [f"{a}_{sample}{ext}" 
                              for a in (alias, alias.lower()) 
                              for ext in exts]
                path_found = None
                for fname in candidates:
                    p = base / fname
                    if p.is_file():            # exact match only
                        path_found = p
                        break
                if path_found is not None:
                    found[tag] = path_found
                    found_in_any_dir = True
                    break  # stop scanning other dirs for this tag
            if found_in_any_dir:
                break  # stop trying other aliases for this tag

    if not found:
        raise FileNotFoundError(f"No UV-Vis files found for sample '{sample}' in {list(search_dirs)}.")

    # ---- Read and collect as Series ----
    series_list = []

    for tag, path in found.items():
        try:
            df = read_uvvis(str(path))             # vendor-aware loader (µm + 0–1 scaling)
            series = df.iloc[:, 0].rename(tag)     # canonical column name = tag
            series_list.append(series)
            
        except Exception as e:
            print(f"Warning: skipping {path.name} ({e})")

    if not series_list:
        raise ValueError(f"Found files for '{sample}', but none could be parsed.")

    # ---- Outer-join on union wavelength grid ----
    out = pd.concat(series_list, axis=1, join="outer").sort_index()

    # ---- Interpolate each column onto the union grid (INTERIOR ONLY) ----
    # This fills NaNs created by mismatched wavelength arrays, but does not extrapolate ends.
    out = out.interpolate(method="index", limit_area="inside")

    # Derive missing components
    out = _derive_family(out, "Rtot", "Rdif", "Rspec")
    out = _derive_family(out, "Ttot", "Tdif", "Tspec")

    # ---- Load standard reflectance and apply correction ---------------------
    if ref_standard is not None:
        out = _correct_reflectance(out, ref_standard)
    
    # set sample name to the dataframe attribute
    out.attrs["sample_name"] = sample

    # Final tidy-up: numeric dtypes and sorted columns
    out = out.apply(pd.to_numeric, errors="coerce").sort_index(axis=1)

    return out

# def rt_style_mapper(sample: pd.DataFrame,
#                 linestyles: Dict = {"tot": "-", "spec": ":", "dif": "--"},
#                 colors: Dict = {"R": "r", "T": "b", "A": "k"}) -> str:
#     """
#     Generate matplotlib line styles for UV-Vis DataFrame columns.
#     E.g. {"Rtot": "-r", "Tspec": ":b", "Rdif": "--r"}.
    
#     Parameters
#     ----------
#     sample : pandas.DataFrame
#         DataFrame with columns named like Rtot, Tspec, Rdif, Atot, etc.
    
#     linestyles : Dict, optional
#         Mapping of line type keywords to matplotlib line styles.
#         Defaults to {"tot": "-", "spec": ":", "dif": "--"}.
    
#     colors : Dict, optional
#         Mapping of measurement type keywords to colors.
#         Defaults to {"R": "r", "T": "b", "A": "k"}.
    
#     Returns
#     -------
#     Dict[str, str]
#         Dictionary mapping each column name to a matplotlib style string.
#         E.g. {"Rtot": "-r", "Tspec": ":b", "Rdif": "--r"}
    
#     Notes
#     -----
#     - The function looks for keywords in column names to determine line style
#       and color. It is case-insensitive.
    
#     - If a column name does not match any known keywords, it defaults to
#       a solid black line ("-k").
#     """
    
#     style = {}
#     label = {}
#     # Case-insensitive matching
#     for col_name in sample.columns:
#         color = colors.get(col_name[0], "k")  # default black if not R/T
        
#         found = False
#         for key, ls in linestyles.items():
#             # match line style keywords at end of column name
#             if col_name.lower().endswith(key):
#                 style[col_name] = ls + color
#                 label[col_name] = f"${col_name[0]}_\mathrm{{{key}}}$"
#                 found = True
#                 break

#         # Fallback solid line if no style keyword matched
#         if not found:
#             style[col_name] = "-" + color
#             label[col_name] = col_name
            
#     return style, label

def _correct_reflectance(out: pd.DataFrame, ref_standard: str) -> None:
    # Path relative to the empylib package
    ref_standard += "" if ref_standard.endswith(".csv") else ".csv"

    ref_path = Path(__file__).parent / "IS_reflectance_reference" / ref_standard
    if not ref_path.is_file():
        raise FileNotFoundError(f"Reflectance standard not found: {ref_path}")

    ref = pd.read_csv(ref_path)
    # Expect first column = wavelength (µm or nm), second = reflectance fraction (0–1)
    ref.columns = [c.strip().lower() for c in ref.columns]
    wl_col = ref.columns[0]
    R_col = ref.columns[1]
    ref = ref.rename(columns={wl_col: "wavelength", R_col: "Rstd"})
    ref = ref.set_index("wavelength")

    # Match wavelength units: if measurement index ~hundreds, assume nm → convert to µm
    if out.index.max() > 100:  # e.g., 200–2500 nm
        out.index = out.index / 1000.0

    # Interpolate standard to same wavelength grid
    Rstd = ref["Rstd"].reindex(out.index, method=None)
    if Rstd.isna().any():
        Rstd = ref["Rstd"].interpolate(method="index").reindex(out.index, method="nearest", fill_value=None)

    # Apply correction only to reflectance columns
    for col in out.columns:
        if col.startswith("R"):
            out[col] = out[col] / Rstd.values

    return out