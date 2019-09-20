from demo import webcam_demo,video_demo,image_demo,get_cmd_args


def main():
    args = get_cmd_args()
    json_path = args.json_path
    weights = args.weights
    image_height = args.image_size[0]
    image_width = args.image_size[1]
    if args.process == "webcam":
        webcam_demo(json_path,weights)
    elif args.process == "video":
        video_path = args.path
        video_demo(json_path,weights,video_path)
    elif args.process == "image":
        image_path = args.path
        image_demo(json_path,weights,image_path,image_height,image_width)
    else:
        print ("Unrecognized argument for --process",args.process)
if __name__ == "__main__":
    main()
