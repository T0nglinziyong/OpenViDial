from model import MyModel
from data_generation import get_data
import pytorch_lightning as pl
from pyinstrument import Profiler
from pytorch_lightning.callbacks import ModelCheckpoint

# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
file_name = 'model-{epoch:02d}-{train_loss:.2f}'

checkpoint_callback = ModelCheckpoint(
    monitor = 'train_loss',
    filename = file_name,
    #save_top_k = 3,
    mode='min',
    save_last=True
)  # 用于保存训练参数


if __name__ == "__main__":
    profiler = Profiler()
    profiler.start()
    dataloader, vocab = get_data(batch_size=32)
    net = MyModel(vocab=vocab)
    net = MyModel.load_from_checkpoint('D:\\程序设计-python\\OpenViDial\\MyOpenViDial\\lightning_logs\\version_0\\checkpoints\\last.ckpt', vocab=vocab)
    trainer = pl.Trainer(max_epochs=3, fast_dev_run=False, accelerator='cpu', callbacks=[checkpoint_callback])
    trainer.fit(net, dataloader)

    profiler.stop()
    profiler.print()
    '''for src_img, src_tokens, tgt_tokens in dataloader:
        print(src_tokens)
        print(net.get_seq_and_position(src_tokens))
        break'''