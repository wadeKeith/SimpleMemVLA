from pathlib import Path

from transformers import set_seed

from simplememvla.training.sft_trainer import SimpleMemVLATrainer


class TrainRunner:
    def __init__(
        self,
        model,
        training_args,
        train_dataset,
        data_collator=None,
        resume_from_checkpoint=False,
        processor=None,
        data_root=None,
    ):
        self.model = model
        self.training_args = training_args
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.resume_from_checkpoint = resume_from_checkpoint
        self.output_dir = Path(training_args.output_dir)

        if training_args.run_name is None:
            training_args.run_name = training_args.output_dir.split("/")[-1]
        set_seed(training_args.seed)

        self.trainer = SimpleMemVLATrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            processing_class=processor,
            data_root=data_root,
        )

    def train(self):
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        self.trainer.save_state()
        self.trainer.save_model(self.training_args.output_dir)
