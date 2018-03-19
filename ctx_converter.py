import argparse

def unite_ctx(ctx_file, dst_file, save_path):
    ctx = open(ctx_file, "r")
    dst = open(dst_file, "r")
    result = open(save_path, "w")

    for four_sentences, last_sentence in zip(ctx, dst):
        four_sentences = four_sentences.split('_eos')
        for sentence in four_sentences:
            result.write(sentence.strip() + "\n")
        result.write(last_sentence)


    ctx.close()
    dst.close()
    result.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ctx_file', required=True)
    parser.add_argument('--dst_file', required=True)
    parser.add_argument('--save_path', default="result.dst")
    args = parser.parse_args()
    unite_ctx(args.ctx_file, args.dst_file, args.save_path)

if __name__ == '__main__':
    main()
    