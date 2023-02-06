from lxml import etree
from typing import List, Tuple


def vxd_creator(
    sizes: Tuple[int, int, int],
    representation: List[Tuple[List[int], List[float], List[float], List[float]]],
    record_history=True,
):
    """
    Writes the extent and CData to XML file.

    Returns:
        The full xml content as a string.
    """

    VXD = etree.Element("VXD")
    Structure = etree.SubElement(VXD, "Structure")
    Structure.set("replace", "VXA.VXC.Structure")
    Structure.set("Compression", "ASCII_READABLE")
    etree.SubElement(Structure, "X_Voxels").text = f"{sizes[0]}"
    etree.SubElement(Structure, "Y_Voxels").text = f"{sizes[1]}"
    etree.SubElement(Structure, "Z_Voxels").text = f"{sizes[2]}"

    Data = etree.SubElement(Structure, "Data")
    if representation[0][1] is not None:
        amplitudes = etree.SubElement(Structure, "Amplitude")
    else:
        amplitudes = None
    if representation[0][2] is not None:
        frequencies = etree.SubElement(Structure, "Frequency")
    else:
        frequencies = None
    if representation[0][3] is not None:
        phase_offsets = etree.SubElement(Structure, "PhaseOffset")
    else:
        phase_offsets = None
    for z in range(sizes[2]):
        material_data = "".join([f"{m}" for m in representation[z][0]])
        etree.SubElement(Data, "Layer").text = etree.CDATA(material_data)

        if representation[z][1] is not None:
            amplitude_data = "".join([f"{p}, " for p in representation[z][1]])
            etree.SubElement(amplitudes, "Layer").text = etree.CDATA(amplitude_data)

        if representation[z][2] is not None:
            frequency_data = "".join([f"{p}, " for p in representation[z][2]])
            etree.SubElement(frequencies, "Layer").text = etree.CDATA(frequency_data)

        if representation[z][3] is not None:
            phase_offset_data = "".join([f"{p}, " for p in representation[z][3]])
            etree.SubElement(phase_offsets, "Layer").text = etree.CDATA(
                phase_offset_data
            )

    if record_history:
        history = etree.SubElement(VXD, "RecordHistory")
        history.set("replace", "VXA.Simulator.RecordHistory")
        etree.SubElement(history, "RecordStepSize").text = "250"
        etree.SubElement(history, "RecordVoxel").text = "1"
        etree.SubElement(history, "RecordLink").text = "0"
        etree.SubElement(history, "RecordFixedVoxels").text = "0"

    return etree.tostring(VXD, pretty_print=True).decode("utf-8")


def get_voxel_positions(result, voxel_size=0.01):
    doc = etree.fromstring(bytes(result, encoding="utf-8"))

    def parse(x):
        y = x.split(";")
        p = []
        for v in y:
            if len(v) > 0:
                p.append([float(q) / voxel_size for q in v.split(",")])
        return p

    initial_positions = doc.xpath("/report/detail/voxel_initial_positions")[0].text
    final_positions = doc.xpath("/report/detail/voxel_final_positions")[0].text
    initial_positions = parse(initial_positions)
    final_positions = parse(final_positions)
    return initial_positions, final_positions
