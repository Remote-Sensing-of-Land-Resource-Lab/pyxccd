"""Keep wildfire detections from 15 March through 30 April in the 2024 example."""

import argparse
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annual_map', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    first = date(2024, 3, 15).timetuple().tm_yday
    last = date(2024, 4, 30).timetuple().tm_yday
    with rasterio.open(args.annual_map) as source:
        tags = source.tags()
        if (source.dtypes != ('uint16',) or source.nodata != 65535
                or tags.get('layout') != 'ASDDD' or tags.get('year') != '2024'):
            raise ValueError('Expected a 2024 UInt16 ASDDD raster with NoData 65535')
        profile = source.profile
        codes = source.read(1)
    days = codes % 1000
    selected = (codes // 10000 == 1) & (days >= first) & (days <= last)
    fire = np.where(selected | (codes == 65535), codes, 0).astype(np.uint16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as folder:
        temporary = Path(folder) / args.output.name
        with rasterio.open(temporary, 'w', **profile) as target:
            target.write(fire, 1)
            target.update_tags(**tags)
            target.update_tags(filter='wildfire detections from 2024-03-15 to 2024-04-30',
                               zero='monitored forest without selected wildfire')
        temporary.replace(args.output)
    print(f'{np.count_nonzero(selected)} wildfire pixels written to {args.output}')


if __name__ == '__main__':
    main()
