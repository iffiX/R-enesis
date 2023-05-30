import numpy as np
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
        material_data = "".join(np.char.mod("%d", representation[z][0]))
        etree.SubElement(Data, "Layer").text = etree.CDATA(material_data)

        if representation[z][1] is not None:
            amplitude_data = ", ".join(np.char.mod("%f", representation[z][1]))
            etree.SubElement(amplitudes, "Layer").text = etree.CDATA(amplitude_data)

        if representation[z][2] is not None:
            frequency_data = ", ".join(np.char.mod("%f", representation[z][2]))
            etree.SubElement(frequencies, "Layer").text = etree.CDATA(frequency_data)

        if representation[z][3] is not None:
            phase_offset_data = ", ".join(np.char.mod("%f", representation[z][3]))
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
    """Note: Unit is voxels"""
    doc = etree.fromstring(bytes(result, encoding="utf-8"))

    def parse(x):
        y = x.split(";")
        p = []
        for v in y:
            if len(v) > 0:
                p.append([float(q) / voxel_size for q in v.split(",")])
        return p

    start_positions = doc.xpath("/report/detail/voxel_start_positions")[0].text
    end_positions = doc.xpath("/report/detail/voxel_end_positions")[0].text
    start_positions = parse(start_positions)
    end_positions = parse(end_positions)
    return start_positions, end_positions


def get_center_of_mass(result, voxel_size=0.01):
    """Note: Unit is voxels"""
    doc = etree.fromstring(bytes(result, encoding="utf-8"))
    start_com = np.array(
        [
            float(doc.xpath("/report/detail/start_center_of_mass/x")[0].text),
            float(doc.xpath("/report/detail/start_center_of_mass/y")[0].text),
            float(doc.xpath("/report/detail/start_center_of_mass/z")[0].text),
        ]
    )
    end_com = np.array(
        [
            float(doc.xpath("/report/detail/end_center_of_mass/x")[0].text),
            float(doc.xpath("/report/detail/end_center_of_mass/y")[0].text),
            float(doc.xpath("/report/detail/end_center_of_mass/z")[0].text),
        ]
    )
    return start_com / voxel_size, end_com / voxel_size
