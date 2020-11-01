def _read_from_stream(
    container: "av.container.Container",
    start_offset: float,
    end_offset: float,
    pts_unit: str,
    stream: "av.stream.Stream",
    stream_name: Dict[str, Optional[Union[int, Tuple[int, ...], List[int]]]],
) -> List["av.frame.Frame"]:
    global _CALLED_TIMES, _GC_COLLECTION_INTERVAL
    _CALLED_TIMES += 1
    if _CALLED_TIMES % _GC_COLLECTION_INTERVAL == _GC_COLLECTION_INTERVAL - 1:
        gc.collect()

    if pts_unit == "sec":
        start_offset = int(math.floor(start_offset * (1 / stream.time_base)))
        if end_offset != float("inf"):
            end_offset = int(math.ceil(end_offset * (1 / stream.time_base)))
    else:
        warnings.warn(
            "The pts_unit 'pts' gives wrong results and will be removed in a "
            + "follow-up version. Please use pts_unit 'sec'."
        )

    frames = {}
    should_buffer = True
    max_buffer_size = 5
    if stream.type == "video":
        # DivX-style packed B-frames can have out-of-order pts (2 frames in a single pkt)
        # so need to buffer some extra frames to sort everything
        # properly
        extradata = stream.codec_context.extradata
        # overly complicated way of finding if `divx_packed` is set, following
        # https://github.com/FFmpeg/FFmpeg/commit/d5a21172283572af587b3d939eba0091484d3263
        if extradata and b"DivX" in extradata:
            # can't use regex directly because of some weird characters sometimes...
            pos = extradata.find(b"DivX")
            d = extradata[pos:]
            o = re.search(br"DivX(\d+)Build(\d+)(\w)", d)
            if o is None:
                o = re.search(br"DivX(\d+)b(\d+)(\w)", d)
            if o is not None:
                should_buffer = o.group(3) == b"p"
    seek_offset = start_offset
    # some files don't seek to the right location, so better be safe here
    seek_offset = max(seek_offset - 1, 0)
    if should_buffer:
        # FIXME this is kind of a hack, but we will jump to the previous keyframe
        # so this will be safe
        seek_offset = max(seek_offset - max_buffer_size, 0)
    try:
        # TODO check if stream needs to always be the video stream here or not
        container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    except av.AVError:
        # TODO add some warnings in this case
        # print("Corrupted file?", container.name)
        return []
    buffer_count = 0
    try:
        for _idx, frame in enumerate(container.decode(**stream_name)):
            frames[frame.pts] = frame
            if frame.pts >= end_offset:
                if should_buffer and buffer_count < max_buffer_size:
                    buffer_count += 1
                    continue
                break
    except av.AVError:
        # TODO add a warning
        pass
    # ensure that the results are sorted wrt the pts
    result = [
        frames[i] for i in sorted(frames) if start_offset <= frames[i].pts <= end_offset
    ]
    if len(frames) > 0 and start_offset > 0 and start_offset not in frames:
        # if there is no frame that exactly matches the pts of start_offset
        # add the last frame smaller than start_offset, to guarantee that
        # we will have all the necessary data. This is most useful for audio
        preceding_frames = [i for i in frames if i < start_offset]
        if len(preceding_frames) > 0:
            first_frame_pts = max(preceding_frames)
            result.insert(0, frames[first_frame_pts])
    return result