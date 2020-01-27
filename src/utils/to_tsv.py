import argparse
import csv

def csv_2_tsv(in_path: str, out_path: str, cols=[]) -> None:
    new_rows = []
    with open(in_path, "r") as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            if cols == []:
                new_rows.append(row)
            else:
                new_rows.append([row[int(col)] for col in cols])
    
    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file, delimiter='\t')
        for row in new_rows:
            writer.writerow(row)
    
def main(args):
    csv_2_tsv(args.in_path, args.out_path, args.cols)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_path", help="The source path of file to be converted to tsv", required=True)
    parser.add_argument("-o", "--out_path", help="The path of the output tsv file", required=True)
    parser.add_argument("-c", "--cols", nargs="+", help="Columns of the input file to write into csv", required=False, default=[])
    args = parser.parse_args()
    main(args)

