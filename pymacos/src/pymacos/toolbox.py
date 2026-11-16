# Script Collection that may be useful

import os
from typing import Tuple, List
import re
from pathlib import Path


def rm_srf_from_rx(rx_in: Path | str,
                   srf_idx: Tuple | List | None = None,
                   rx_out: Path | str | None = None) -> None:
    """Eliminate surface(s) from Rx.

    Args:
        rx_in (Path | str):
            File (path & name) of file to be processed

        srf_idx (Tuple | List | None, optional):
            - if not defined (None), it will list all surfaces in Rx
            - if Srf. Indices are defined, e.g. (1,2), it will remove
              those from the Rx

        rx_out (Path | str | None, optional):
            - if not defined (None), it will list the upd. Rx
            - if a File is provided, it will save the new Rx into
              that file whether it exists or not.

    Important:
        If any references to a surface is given like via "EltGrp",
        those surface references will _not_ be updated.

    Examples:
        list Srf. in Rx
            rm_srf_from_rx('./Rx/my_Rx.in')

        remove Srf #5 and list the upd. Rx
            rm_srf_from_rx('./Rx/my_Rx.in', (5,))

        remove Srf #5 and save the Rx as new file: 'my_Rx_stripped.in'
            rm_srf_from_rx('./Rx/my_Rx.in', (5,), 'my_Rx_stripped.in')
    """

    def list_surfaces(surfaces:dict[int, str]):
        """ Srf. Listing with parameters: iElt, EltName and Element.
        """
        lines = []
        for index in surfaces.keys():
            block = surfaces[index]

            # Extract EltName and Element with flexible whitespace
            elt_name_match = re.search(r"EltName\s*=\s*([^\s]+)", block)
            element_match = re.search(r"Element\s*=\s*([^\s]+)", block)

            elt_name = elt_name_match.group(1) if elt_name_match else "Unknown"
            element = element_match.group(1) if element_match else "Unknown"

            lines.append("".join((f"Srf. {index:03}: ",
                                  f"iElt= {index:3}, ",
                                  f"EltName= {elt_name:14}, ",
                                  f"Element= {element:10}")))
        return "\n".join(lines)

    def rx_split(data: str) -> Tuple[str, dict[int, str], str]:
        """MACOS Rx is split into Header, Surfaces & Footer
        """
        pattern = (r"(?P<indent>\s*)iElt\s*=\s*(\d+)"
                   r"(?P<body>[\s\S]*?)(?=\n\s*iElt\s*=|\n\s*nOutCord|\Z)")

        srf = []
        for m in re.finditer(pattern, data):
            indent = m.group("indent")
            index = int(m.group(2))
            block = indent + "iElt= " + m.group(2) + m.group("body")
            start = m.start()
            end = m.end()
            srf.append((index, block, start, end))

        # extract Header & Footer
        first_start = srf[0][2]
        last_end = srf[-1][3]

        header = data[:first_start]
        footer = data[last_end:]
        surfaces = reindex_surfaces({idx: block for (idx, block, __, __) in srf})

        return header, surfaces, footer

    def reindex_surfaces(surfaces:dict[int, str]) -> dict[int, str]:
        """iElt in Srf. Block is updated to ensure a sequential numbering
        """
        reordered = {}
        for new_idx, (old_idx, block) in enumerate(surfaces.items(), start=1):
            block = re.sub(r"(^\s*)iElt\s*=\s*\d+", r"\1iElt= {}".format(new_idx), block, count=1)
            reordered[new_idx] = block
        return reordered

    # start process

    if not Path(rx_in).exists():
        raise FileNotFoundError(f"{rx_in} does not exist")

    if rx_out is not None:
        rx_out = Path(fr"{rx_out}")
        if (not rx_out.parent.exists() or
                not os.access(rx_out.parent, os.W_OK)):
            raise FileNotFoundError(f"access issues to dir '{rx_out.parent}'")

    # load Rx as string
    with open(rx_in, "r") as f:
        data = f.read()

    # separate header, body & footer
    header, surfaces, footer = rx_split(data)

    if srf_idx is None:
        print(list_surfaces(surfaces))
        return

    # expunge surfaces
    if not set(srf_idx).issubset(set(surfaces.keys())):
        raise ValueError(f"Not all indices in {srf_idx} exist")

    if set(surfaces.keys()) == set(srf_idx):
        raise ValueError(f"Cannot remove ALL Surfaces")

    surfaces = reindex_surfaces({k: v for k, v in surfaces.items()
                                 if k not in srf_idx})

    # body = "".join(surfaces[i] for i in surfaces)
    body = "".join(surfaces.values())

    # update header with new nElt value
    n_srf = len(surfaces)

    # Regex to find the nElt tag, preserving the indentation before it
    match = re.search(r"(\s*)nElt\s*=\s*\d+", header)
    if match:
        indentation = match.group(1)  # Capture the indentation
        header = re.sub(r"(\s*)nElt\s*=\s*\d+",
                        f"{indentation}nElt= {n_srf}", header)

    # Reconstruct the entire file with header, updated surfaces, and footer
    data = f"{header}\n{body}\n{footer}"
    # data = rebuild_file(header, surfaces, footer)

    if rx_out is None:
        print(data)
    else:
        with open(rx_out, "w") as f:
            f.write(data)

