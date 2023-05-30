import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from subprocess import Popen
from renesis.utils.robot import (
    get_robot_voxels_from_voxels,
    get_representation_from_robot_voxels,
)
from renesis.utils.voxcraft import vxd_creator, get_voxel_positions, get_center_of_mass
from renesis.utils.metrics import distance_traveled
from renesis.sim import Voxcraft
from experiments.navigator.trial import TrialRecord
from experiments.navigator.prompter import PromptInputException

DEFAULT_VXA = """

<VXA Version="1.1">
    <GPU>
        <HeapSize>0.5</HeapSize>
    </GPU>
    <Simulator>
        <EnableExpansion>1</EnableExpansion>
        <Integration>
            <DtFrac>1
            </DtFrac>
        </Integration>
        <Condition>
            <StopCondition>
                <mtSUB>
                <mtVAR>t</mtVAR>
                <mtCONST>10</mtCONST>
                </mtSUB>
            </StopCondition>
            <ResultStartCondition>
                <mtSUB>
                <mtVAR>t</mtVAR>
                <mtCONST>5</mtCONST>
                </mtSUB>
            </ResultStartCondition>
            <ResultEndCondition>
                <mtSUB>
                <mtVAR>t</mtVAR>
                <mtCONST>10</mtCONST>
                </mtSUB>
            </ResultEndCondition>
        </Condition>
        <SavePositionOfAllVoxels>1</SavePositionOfAllVoxels>
        <Damping>
            <BondDampingZ>1</BondDampingZ>
            <ColDampingZ>0.8</ColDampingZ>
            <SlowDampingZ>0.03</SlowDampingZ>
        </Damping>
    </Simulator>
    <Environment>
        <Thermal>
            <TempEnabled>1</TempEnabled>
            <TempAmplitude>10</TempAmplitude>
            <VaryTempEnabled>1</VaryTempEnabled>
            <TempPeriod>.25</TempPeriod>
        </Thermal>
        <Gravity>
            <GravEnabled>1</GravEnabled>
            <GravAcc>-9.81</GravAcc>
            <FloorEnabled>1</FloorEnabled>
        </Gravity>
    </Environment>
<VXC Version="0.94">
    <Lattice>
        <Lattice_Dim>0.01</Lattice_Dim>
    </Lattice>
    <Palette>
        <Material ID="1">
            <Name>Body</Name>
            <Display>
                <Red>0</Red>
                <Green>0</Green>
                <Blue>1</Blue>
                <Alpha>0.75</Alpha>
            </Display>
            <Mechanical>
                <MatModel>0</MatModel><!--0 = no failing-->
                <Elastic_Mod>1e5</Elastic_Mod>
                <Fail_Stress>0</Fail_Stress>
                <Density>1500</Density>
                <Poissons_Ratio>0.35</Poissons_Ratio>
                <CTE>0</CTE>
                <MaterialTempPhase>0</MaterialTempPhase>
                <uStatic>1</uStatic>
                <uDynamic>0.5</uDynamic>
            </Mechanical>
        </Material>
        <Material ID="2">
            <Name>Motor1</Name>
            <Display>
                <Red>0</Red>
                <Green>1</Green>
                <Blue>0</Blue>
                <Alpha>0.75</Alpha>
            </Display>
            <Mechanical>
                <MatModel>0</MatModel><!--0 = no failing-->
                <Elastic_Mod>1e5</Elastic_Mod>
                <Fail_Stress>0</Fail_Stress>
                <Density>1500</Density>
                <Poissons_Ratio>0.35</Poissons_Ratio>
                <CTE>0.01</CTE>
                <MaterialTempPhase>0</MaterialTempPhase>
                <uStatic>1</uStatic>
                <uDynamic>0.5</uDynamic>
            </Mechanical>
        </Material>
        <Material ID="3">
            <Name>Motor2</Name>
            <Display>
                <Red>1</Red>
                <Green>0</Green>
                <Blue>0</Blue>
                <Alpha>0.75</Alpha>
            </Display>
            <Mechanical>
                <MatModel>0</MatModel><!--0 = no failing-->
                <Elastic_Mod>1e5</Elastic_Mod>
                <Fail_Stress>0</Fail_Stress>
                <Density>1500</Density>
                <Poissons_Ratio>0.35</Poissons_Ratio>
                <CTE>0.01</CTE>
                <MaterialTempPhase>0.5</MaterialTempPhase>
                <uStatic>1</uStatic>
                <uDynamic>0.5</uDynamic>
            </Mechanical>
        </Material>
    </Palette>
    <Structure Compression="ASCII_READABLE">
        <X_Voxels>2</X_Voxels>
        <Y_Voxels>2</Y_Voxels>
        <Z_Voxels>3</Z_Voxels>
        <Data>
        </Data>
    </Structure>
</VXC>
</VXA>"""


def visualize_robot(trial_record: TrialRecord, show_epoch: int = -1):
    if show_epoch not in trial_record.epochs:
        if show_epoch > 0:
            print(f"Required epoch {show_epoch} not found")
        print("Use epoch with max reward")
        show_epoch = trial_record.max_reward_epoch
    command = ["voxcraft-viz", trial_record.epoch_files[show_epoch].history_file_name]
    print(" ".join(command))
    process = Popen(command, cwd=trial_record.data_dir)
    process.wait()


def visualize_selected_robot(trial_record: TrialRecord, show_epoch: int = -1):
    if show_epoch not in trial_record.epochs:
        if show_epoch > 0:
            print(f"Required epoch {show_epoch} not found")
        print("Use epoch with max reward")
        show_epoch = trial_record.max_reward_epoch

    with open(
        os.path.join(
            trial_record.data_dir, trial_record.epoch_files[show_epoch].data_file_name
        ),
        "rb",
    ) as file:
        data = pickle.load(file)
        data = sorted(data, key=lambda d: d["reward"], reverse=True)
        row_size = int(np.ceil(np.sqrt(len(data) / 2)))
        col_size = row_size * 2

        fig, axs = plt.subplots(row_size, col_size, subplot_kw={"projection": "3d"})

        for row in range(row_size):
            for col in range(col_size):
                idx = row * col_size + col

                if idx < len(data):
                    if data[idx]["reward"] <= 0:
                        axs[row][col].set_facecolor([0.2, 0.2, 0.2])
                    axs[row][col].set_title(f"{idx}:{data[idx]['reward']:.3f}")
                    axs[row][col].set_xticks([])
                    axs[row][col].set_yticks([])
                    axs[row][col].set_zticks([])

                    robot_voxels, robot_occupied = get_robot_voxels_from_voxels(
                        data[idx]["voxels"]
                    )
                    colors = np.empty(robot_voxels.shape, dtype=object)
                    colors[robot_voxels == 1] = "blue"
                    colors[robot_voxels == 2] = "green"
                    colors[robot_voxels == 3] = "red"
                    axs[row][col].voxels(robot_occupied, facecolors=colors)
                    axs[row][col].axis("equal")
        try:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        except:
            pass
        fig.set_figheight(4.5 * row_size)
        fig.set_figwidth(4.5 * col_size)
        fig.show()

    while True:
        try:
            selected_idx = input("Select robot index to visualize, Ctrl-C to break: ")
            selected_idx = int(selected_idx)
        except KeyboardInterrupt:
            return
        except:
            raise PromptInputException()

        os.makedirs(
            os.path.join("visualize_data", trial_record.trial_dir.replace("/", "#")),
            exist_ok=True,
        )
        with open(
            os.path.join(
                "visualize_data",
                trial_record.trial_dir.replace("/", "#"),
                f"tmp_it_{show_epoch}_robot_{selected_idx}_rew_{data[selected_idx]['reward']}.history",
            ),
            "w",
        ) as file:
            robot_voxels, _ = get_robot_voxels_from_voxels(data[selected_idx]["voxels"])
            vxd = vxd_creator(*get_representation_from_robot_voxels(robot_voxels))
            if not trial_record.vxa_file:
                print("VXA file not found, using default VXA settings")
                vxa = DEFAULT_VXA
            else:
                with open(trial_record.vxa_file, "r") as vxa_file:
                    vxa = vxa_file.read()

            simulator = Voxcraft()
            begin = time.time()
            results, records = simulator.run_sims([vxa], [vxd])
            end = time.time()
            print(f"Simulation time {end - begin:.3f}")
            file.write(records[0])

            start_com, end_com = get_center_of_mass(results[0], voxel_size=0.01)
            print(distance_traveled(start_com, end_com))
        command = [
            "voxcraft-viz",
            file.name,
        ]
        print(" ".join(command))
        process = Popen(command)
        process.wait()
