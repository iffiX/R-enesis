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
    amplitudes = etree.SubElement(Structure, "Amplitude")
    frequencies = etree.SubElement(Structure, "Frequency")
    phase_offsets = etree.SubElement(Structure, "PhaseOffset")
    for z in range(sizes[2]):
        material_data = "".join([f"{m}" for m in representation[z][0]])
        amplitude_data = "".join([f"{p}, " for p in representation[z][1]])
        frequency_data = "".join([f"{p}, " for p in representation[z][2]])
        phase_offset_data = "".join([f"{p}, " for p in representation[z][3]])
        etree.SubElement(Data, "Layer").text = etree.CDATA(material_data)
        etree.SubElement(amplitudes, "Layer").text = etree.CDATA(amplitude_data)
        etree.SubElement(frequencies, "Layer").text = etree.CDATA(frequency_data)
        etree.SubElement(phase_offsets, "Layer").text = etree.CDATA(phase_offset_data)

    if record_history:
        history = etree.SubElement(VXD, "RecordHistory")
        history.set("replace", "VXA.Simulator.RecordHistory")
        etree.SubElement(history, "RecordStepSize").text = "250"
        etree.SubElement(history, "RecordVoxel").text = "1"
        etree.SubElement(history, "RecordLink").text = "0"
        etree.SubElement(history, "RecordFixedVoxels").text = "0"

    return etree.tostring(VXD, pretty_print=True).decode("utf-8")


def get_voxel_positions(out_file_path, voxel_size=0.01):
    doc = etree.parse(out_file_path)

    def parse(x):
        y = x.split(";")
        p = []
        for v in y:
            if len(v) > 0:
                p.append([float(q) / voxel_size for q in v.split(",")])
        return p

    initial_positions = doc.xpath("/report/detail/robot/init_pos")[0].text
    final_positions = doc.xpath("/report/detail/robot/pos")[0].text
    return parse(initial_positions), parse(final_positions)
