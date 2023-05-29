import re
from typing import List, Callable, Tuple, Any


class PromptInputException(Exception):
    pass


class InputType:
    def __init__(self, input_type, optional=False):
        self.input_type = input_type
        self.optional = optional

    def convert(self, value: str):
        raise NotImplementedError()


class Int(InputType):
    def __init__(self, optional=False):
        super(Int, self).__init__(int, optional=optional)

    def convert(self, value):
        try:
            return int(value)
        except ValueError:
            print("Input format incorrect")
            raise PromptInputException()


class Float(InputType):
    def __init__(self, optional=False):
        super(Float, self).__init__(int, optional=optional)

    def convert(self, value):
        try:
            return float(value)
        except ValueError:
            print("Input format incorrect")
            raise PromptInputException()


class String(InputType):
    def __init__(self, optional=False):
        super(String, self).__init__(int, optional=optional)

    def convert(self, value):
        return value


class PromptStage:
    def __init__(self, description):
        self.description = description

    def run(self, stack):
        raise NotImplementedError()


class PromptExecutable(PromptStage):
    def __init__(self, description: str, execute: Callable):
        super(PromptExecutable, self).__init__(description)
        self.execute = execute

    def run(self, stack):
        self.execute()


class PromptExecutableWithInput(PromptExecutable):
    def __init__(
        self,
        description: str,
        execute: Callable,
        prompt_input: "Input parameters:",
        input_formats: List[InputType] = None,
        input_converter: Callable[[str], tuple] = None,
    ):
        super(PromptExecutableWithInput, self).__init__(description, execute)
        self.prompt_input = prompt_input
        if (input_formats is not None and input_converter is not None) or (
            input_formats is None and input_converter is None
        ):
            raise ValueError(
                "Specify either input_formats or input_converter, and not both"
            )
        self.input_formats = input_formats
        self.input_converter = input_converter

    def run(self, stack):
        raw_params = input(self.prompt_input)
        if self.input_converter:
            self.execute(*self.input_converter(raw_params))
        else:
            params = []
            for raw_param, format in zip(raw_params.split(","), self.input_formats):
                params.append(format.convert(raw_param))
            self.execute(*params)


class PromptExecutableWithMultipleChoice(PromptStage):
    def __init__(
        self,
        description: str,
        execute: Callable,
        choices: List[Tuple[str, Any]],
        prompt_title: str = "Options:",
        prompt_input: str = "Select multiple choices, split by comma:",
    ):
        super(PromptExecutableWithMultipleChoice, self).__init__(description)
        self.execute = execute
        self.choices = choices
        self.prompt_title = prompt_title
        self.prompt_input = prompt_input

    def run(self, stack):
        stack.append(self)
        print(self.prompt_title)
        for idx, choice in enumerate(self.choices):
            print(f"{idx}) {choice[0]}")
        choice_indices = input(self.prompt_input)
        try:
            choice_indices = [
                int(choice_idx) for choice_idx in choice_indices.split(",")
            ]
        except:
            print(
                f"{choice_indices} is not a valid input, input multiple indices, separated by comma"
            )
            raise PromptInputException()
        self.execute([self.choices[idx][1] for idx in choice_indices])


class PromptChoiceDialog(PromptStage):
    def __init__(
        self,
        description: str,
        choices: List[PromptStage],
        prompt_title: str = "Options:",
        prompt_input: str = "Choice:",
    ):
        super(PromptChoiceDialog, self).__init__(description)
        self.choices = choices
        self.prompt_title = prompt_title
        self.prompt_input = prompt_input

    def run(self, stack):
        stack.append(self)
        print(self.prompt_title)
        for idx, choice in enumerate(self.choices):
            print(f"{idx}) {choice.description}")
        choice_idx = input(self.prompt_input)
        try:
            choice_idx = int(choice_idx)
            choice = self.choices[choice_idx]
        except:
            print(f"{choice_idx} is not a valid choice index")
            raise PromptInputException()
        choice.run(stack)


class PromptApp:
    def __init__(self, root_prompt: PromptChoiceDialog):
        self.root_prompt = root_prompt

    def run(self):
        while True:
            print(">" * 100)
            stack = []
            try:
                self.root_prompt.run(stack)
            except PromptInputException:
                continue
            except KeyboardInterrupt:
                print("Cancel running function")
                if input("Continue? [Y/N]").lower() in ("y", "yes"):
                    continue
                else:
                    break
