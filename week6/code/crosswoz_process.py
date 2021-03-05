import json
from tqdm import tqdm
import os
import random
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from functools import partial
MAX_SEQ_LEN = 512

class MyDataset:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.ontology = json.load(
            open('data/crosswoz/ontology.json', "r", encoding="utf8"))
        self.train_dataloader = self.load_data(
            data_path='data/crosswoz/val.json', data_type="train"
        )
        self.eval_dataloader = self.load_data(
            data_path='data/crosswoz/val.json', data_type="dev"
        )
        self.test_dataloader = self.load_data(
            data_path='data/crosswoz/test.json', data_type="tests"
        )

    def load_data(self, data_path: str, data_type: str):
        """Loading data by loading cache data or generating examples from scratch.

        Args:
            data_path: raw dialogue data
            data_type: train, dev or tests

        Returns:
            dataloader, see torch.utils.data.DataLoader,
            https://pytorch.org/docs/stable/data.html?highlight=torch%20utils%20data%20dataloader#torch.utils.data.DataLoader
        """
        print(f"Starting preprocess {data_type} data ...")
        raw_data_name = os.path.basename(data_path)
        processed_data_name = "processed_" + raw_data_name.split(".")[0] + ".pt"
        data_cache_path = os.path.join(os.path.dirname(data_path), processed_data_name)
        examples = self.build_examples(data_path, data_cache_path, data_type)

        dataset = DSTDataset(examples)
        shuffle = True if data_type == "train" else False
        collate = partial(collate_fn, mode=data_type)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=32,
            shuffle=shuffle,
            collate_fn=collate
        )
        return dataloader

    def build_examples(
            self, data_path: str, data_cache_path: str, data_type: str
    ):
        """Generate data_type dataset and cache them.

        Args:
            data_path: raw dialogue data path
            data_cache_path: data save path
            data_type: train, dev or tests

        Returns:
            examples, mix up positive and negative examples
        """
        dials = json.load(open(data_path, "r", encoding="utf8"))
        dials = list(dials.items())
        ##########################
        debug = True
        if debug:
            dials = dials[:20]
        ##########################
        pos_examples, neg_examples = self.iter_dials(dials, data_type)

        examples = pos_examples + neg_examples
        print(f"{len(dials)} dialogs generate {len(examples)} examples ...")

        random.shuffle(examples)
        examples = list(zip(*examples))
        torch.save(examples, data_cache_path)

        return examples

    def iter_dials(
            self,
            dials,
            data_type
    ) :
        """Iterate on dialogues, turns in one dialogue to generate examples

        Args:
            dials: raw dialogues data
            data_type: train, dev or tests
        """
        pos_examples = []
        neg_examples = []
        for dial_id, dial in tqdm(
                dials, desc=f"Building {data_type} examples"
        ):
            sys_utter = "对话开始"
            for turn_id, turn in enumerate(dial["messages"]):
                if turn["role"] == "sys":
                    sys_utter = turn["content"]
                else:
                    raw_user_state = turn["user_state"]
                    user_state = self.format_user_state(raw_user_state)

                    usr_utter = turn["content"]
                    context = sys_utter + self.tokenizer.sep_token + usr_utter
                    context_ids = self.tokenizer.encode(
                        context, add_special_tokens=False
                    )

                    cur_dialog_act = turn["dialog_act"]
                    triple_labels = self.format_labels(cur_dialog_act)

                    cur_pos_examples, cur_neg_examples = self.ontology2examples(
                        user_state, context_ids, dial_id, triple_labels, turn_id
                    )

                    pos_examples.extend(cur_pos_examples)
                    neg_examples.extend(cur_neg_examples)
        return pos_examples, neg_examples

    @staticmethod
    def format_user_state(
            raw_user_state):

        """Reformat raw user state to the format that model need.
        Returns:
            user_state, reformatted user state
        """
        user_state = []
        for bs in raw_user_state:
            domain, slot, value = bs[1], bs[2], bs[3]
            if '-' in slot:
                slot, value = slot.split("-")
            user_state.append((domain, slot, value))
        return user_state

    @staticmethod
    def format_labels(dialog_act):
        """Reformat raw dialog act to triple labels.

        Args:
            dialog_act: [
                          [
                            "Inform",
                            "餐馆",
                            "周边景点",
                            "小汤山现代农业科技示范园"
                          ],...
                        ]

        Returns:
            triple_labels, reformatted labels, (domain, slot, value)
        """
        turn_labels = dialog_act
        triple_labels = set()
        for usr_da in turn_labels:
            intent, domain, slot, value = usr_da
            if intent == "Request":
                triple_labels.add((domain, "Request", slot))
            else:
                if "-" in slot:  # 酒店设施
                    slot, value = slot.split("-")
                triple_labels.add((domain, slot, value))
        return triple_labels

    def ontology2examples(
            self,
            user_state,
            context_ids,
            dial_id,
            triple_labels,
            turn_id,
    ):
        """Iterate item in ontology to build examples.

        Args:
            user_state: return value of method `format_user_state`
            context_ids: context token's id in bert vocab
            dial_id: dialogue id in raw dialogue data
            triple_labels: triple label (domain, slot, value)
            turn_id: turn id in one dialogue

        Returns:
            new generated examples based on ontology
        """
        pos_examples = []
        neg_examples = []

        for (
                domain_slots,
                values,
        ) in self.ontology.items():
            domain_slot = domain_slots.split("-")
            domain, slot = domain_slot

            if domain in ["reqmore"]:
                continue

            if domain not in ["greet", "welcome", "thank", "bye"] and slot != "酒店设施":
                example = turn2example(
                    self.tokenizer,
                    domain,
                    "Request",
                    slot,
                    context_ids,
                    triple_labels,
                    user_state,
                    dial_id,
                    turn_id,
                )
                self.get_pos_neg_examples(example, pos_examples, neg_examples)
            value_num=len(values)
            if value_num>10:
                index_list=random.sample(range(value_num),10)
            else:
                index_list=range(value_num)
            for index in index_list:
                value=values[index]
                value = "".join(value.split(" "))
                if slot == "酒店设施":
                    slot_value = slot + f"-{value}"
                    example = turn2example(
                        self.tokenizer,
                        domain,
                        "Request",
                        slot_value,
                        context_ids,
                        triple_labels,
                        user_state,
                        dial_id,
                        turn_id,
                    )
                    self.get_pos_neg_examples(example, pos_examples, neg_examples)

                example = turn2example(
                    self.tokenizer,
                    domain,
                    slot,
                    value,
                    context_ids,
                    triple_labels,
                    user_state,
                    dial_id,
                    turn_id,
                )

                self.get_pos_neg_examples(example, pos_examples, neg_examples)

        neg_examples = random.sample(
            neg_examples,
            k=int(0.9 * len(pos_examples))
        )

        return pos_examples, neg_examples
    @staticmethod
    def get_pos_neg_examples(
        example: tuple, pos_examples, neg_examples
    ) -> None:
        """According to label saving example.

        Args:
            example: a new generated example, (input_ids, token_type_ids, domain, slot, value ...)
            pos_examples: all positive examples are saved in pos_examples
            neg_examples: all negative examples are saved in pos_examples
        """
        if example[-3] == 1:
            pos_examples.append(example)
        else:
            neg_examples.append(example)


def turn2example(
        tokenizer,
        domain: str,
        slot: str,
        value: str,
        context_ids,
        triple_labels=None,
        belief_state=None,
        dial_id: str = None,
        turn_id: int = None):
    """Convert turn data to example based on ontology.

    Args:
        tokenizer: BertTokenizer, see https://huggingface.co/transformers/model_doc/bert.html#berttokenizer
        domain: domain of current example
        slot: slot of current example
        value: value of current example
        context_ids: context token's id in bert vocab
        triple_labels: set of (domain, slot, value)
        belief_state: list of (domain, slot, value)
        dial_id: current dialogue id
        turn_id: current turn id

    Returns:
        example, (input_ids, token_type_ids, domain, slot, value, ...)
    """
    candidate = domain + "-" + slot + " = " + value
    candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
    input_ids = (
            [tokenizer.cls_token_id]
            + context_ids
            + [tokenizer.sep_token_id]
            + candidate_ids
            + [tokenizer.sep_token_id]
    )
    token_type_ids = [0] + [0] * len(context_ids) + [0] + [1] * len(candidate_ids) + [1]
    example = (input_ids, token_type_ids, domain, slot, value)
    if dial_id is not None:
        label = int((domain, slot, value) in triple_labels)
        example += (belief_state, label, dial_id, str(turn_id))
    return example

class DSTDataset(Dataset):
    def __init__(self, examples):
        super(DSTDataset, self).__init__()
        if len(examples) > 5:
            (
                self.input_ids,
                self.token_type_ids,
                self.domains,
                self.slots,
                self.values,
                self.belief_states,
                self.labels,
                self.dialogue_idxs,
                self.turn_ids,
            ) = examples
        else:
            (
                self.input_ids,
                self.token_type_ids,
                self.domains,
                self.slots,
                self.values,
            ) = examples

    def __getitem__(self, index: int) -> tuple:
        if hasattr(self, "labels"):
            return (
                self.input_ids[index],
                self.token_type_ids[index],
                self.domains[index],
                self.slots[index],
                self.values[index],
                self.belief_states[index],
                self.labels[index],
                self.dialogue_idxs[index],
                self.turn_ids[index],
            )
        else:
            return (
                self.input_ids[index],
                self.token_type_ids[index],
                self.domains[index],
                self.slots[index],
                self.values[index],
            )

    def __len__(self):
        return len(self.input_ids)
def collate_fn(examples, mode: str = "train") -> dict:
    """Merge a list of samples to form a mini-batch of Tensor(s)

    generate input_id tensor, token_type_id tensor, attention_mask tensor, pad all tensor to the longest
    sequence in the batch.

    Args:
        examples: list of (input_ids, token_type_ids, domain, slot, value, ...)
        mode: train, dev, tests, infer

    Returns:
        batch data
    """
    batch_examples = {}
    examples = list(zip(*examples))

    if mode == "infer":
        (
            batch_examples["domains"],
            batch_examples["slots"],
            batch_examples["values"],
        ) = examples[2:]
    else:
        (
            batch_examples["domains"],
            batch_examples["slots"],
            batch_examples["values"],
            batch_examples["belief_states"],
            batch_examples["labels"],
            batch_examples["dialogue_idxs"],
            batch_examples["turn_ids"],
        ) = examples[2:]

    attention_mask, input_ids_tensor, token_type_ids_tensor = get_bert_input(examples)

    data = {
        "input_ids": input_ids_tensor,
        "token_type_ids": token_type_ids_tensor,
        "attention_mask": attention_mask,
    }

    if mode == "infer":
        data.update(
            {
                "domains": batch_examples["domains"],
                "slots": batch_examples["slots"],
                "values": batch_examples["values"],
            }
        )
        return data
    else:
        data.update(
            {"labels": torch.tensor(batch_examples["labels"], dtype=torch.long)}
        )

    if mode != "train":
        data.update(
            {
                "domains": batch_examples["domains"],
                "slots": batch_examples["slots"],
                "values": batch_examples["values"],
                "belief_states": batch_examples["belief_states"],
                "dialogue_idxs": batch_examples["dialogue_idxs"],
                "turn_ids": batch_examples["turn_ids"],
            }
        )

    return data
def get_bert_input(
    examples,
) :
    """Convert input list to torch tensor.

    Args:
        examples: (input_id_list, )

    Returns:
        attention_mask, input_ids_tensor, token_type_ids_tensor
    """
    input_ids = examples[0]
    token_type_ids = examples[1]
    max_seq_len = min(max(len(input_id) for input_id in input_ids), MAX_SEQ_LEN)
    input_ids_tensor = torch.zeros((len(input_ids), max_seq_len), dtype=torch.long)
    token_type_ids_tensor = torch.zeros_like(input_ids_tensor)
    attention_mask = torch.ones_like(input_ids_tensor)

    for i, input_id in enumerate(input_ids):
        cur_seq_len = len(input_id)
        if cur_seq_len <= max_seq_len:
            input_ids_tensor[i, :cur_seq_len] = torch.tensor(input_id, dtype=torch.long)
            token_type_ids_tensor[i, :cur_seq_len] = torch.tensor(
                token_type_ids[i], dtype=torch.long
            )
            attention_mask[i, cur_seq_len:] = 0
        else:
            input_ids_tensor[i] = torch.tensor(
                input_id[: max_seq_len - 1] + [102], dtype=torch.long
            )
            token_type_ids_tensor[i] = torch.tensor(
                token_type_ids[i][:max_seq_len], dtype=torch.long
            )

    return attention_mask, input_ids_tensor, token_type_ids_tensor















if __name__ == '__main__':
    test = MyDataset()

