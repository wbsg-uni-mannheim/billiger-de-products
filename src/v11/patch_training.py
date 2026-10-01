"""Apply narrowly checked checkpoint/output fixes to an isolated server copy."""
from pathlib import Path
import argparse


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one patch site: {old[:90]!r}')
    return text.replace(old, new, 1)


def patch(root):
    path = root / 'src/models/hiergat/train.py'
    text = path.read_text()
    text = replace_once(text, '    # start training\n', '''    def save_selected(epoch_number, score):
        if args.save_model:
            checkpoint_dir = Path(args.logdir) / run_tag
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_dir / 'model.pt.tmp'
            torch.save({'model': model.state_dict(), 'epoch': epoch_number,
                        'validation_f1': score, 'args': vars(args)}, temporary)
            os.replace(temporary, checkpoint_dir / 'model.pt')

    # start training
''')
    text = replace_once(text, '\n                best_dev_f1 = dev_f1\n', '''
                best_dev_f1 = dev_f1
                save_selected(epoch - 1, dev_f1)
''')
    text = replace_once(text, '        if dev_f1 > 1e-6:\n', '''        # A collapsed seed is still a result and must have recoverable weights.
        if best_test_preds is None and dev_f1 <= 1e-6:
            save_selected(epoch, dev_f1)
            best_test_preds, best_test_preds_050, best_test_preds_100 = test_preds, test_preds_050, test_preds_100
            best_test_f1, best_test_f1_050, best_test_f1_100 = test_f1, test_f1_050, test_f1_100
        if dev_f1 > 1e-6:
''')
    text = replace_once(text, '    args = parser.parse_args()\n', '''    parser.add_argument('--config_file', default='src/models/hiergat/task.json')
    args = parser.parse_args()
''')
    text = replace_once(text, "json.load(open('src/models/hiergat/task.json'))", 'json.load(open(args.config_file))')
    text = replace_once(text, "((\"000un\", config['testset']), (\"050un\", config['testset050']), (\"100un\", config['testset100']))", '(("000un", testset), ("050un", testset050), ("100un", testset100))')
    path.write_text(text)
    path = root / 'src/models/ditto/train_ditto.py'
    text = path.read_text()
    text = replace_once(text, '    hp = parser.parse_args()\n', '''    parser.add_argument('--config_file', default='src/models/ditto/configs.json')
    hp = parser.parse_args()
''')
    text = replace_once(text, "json.load(open('src/models/ditto/configs.json'))", 'json.load(open(hp.config_file))')
    text = replace_once(text, "((\"000un\", config['testset']), (\"050un\", config['testset050']), (\"100un\", config['testset100']))", '(("000un", testset), ("050un", testset050), ("100un", testset100))')
    path.write_text(text)
    path = root / 'src/models/ditto/ditto_light/ditto.py'
    text = path.read_text()
    text = replace_once(text, '    best_dev_f1 = best_test_f1 = best_test_f1_050 = best_test_f1_100 = 0.0', '    best_dev_f1 = -1.0\n    best_test_f1 = best_test_f1_050 = best_test_f1_100 = 0.0')
    text = replace_once(text, "                        'epoch': epoch}", "                        'epoch': epoch, 'threshold': th,\n                        'validation_f1': dev_f1, 'args': vars(hp)}")
    text = replace_once(text, '                torch.save(ckpt, ckpt_path)', "                torch.save(ckpt, ckpt_path + '.tmp')\n                os.replace(ckpt_path + '.tmp', ckpt_path)")
    path.write_text(text)
    print('Patched HierGAT/Ditto checkpoint saving, config selection and cross-language prediction sources.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    patch(parser.parse_args().root)
