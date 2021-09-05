import argparse
import ast
import csv
import pdb

def csv_2_tsv(in_path: str, out_path: str, cols=[], replace_bracket: bool = True, vec: bool = False) -> None:
    new_rows = []
    with open(in_path, "r") as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            if cols == []:
                new_rows.append(row)
            else:
                if vec:
                    as_string = ",".join(row[int(cols[0]):])
                    vec_lst = ast.literal_eval(as_string)
                    new_rows.append(vec_lst)
                else:
                    new_rows.append([row[int(col)] for col in cols])

    if not vec:
        new_rows.insert(0, ["Year", "Title"])

    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file, delimiter='\t')
        for row in new_rows:
            writer.writerow(row)
    

def main(args):
    csv_2_tsv(args.in_path, args.out_path, args.cols, vec=args.v)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_path", help="The source path of file to be converted to tsv", required=True)
    parser.add_argument("-o", "--out_path", help="The path of the output tsv file", required=True)
    parser.add_argument("-c", "--cols", nargs="+", help="Columns of the input file to write into csv", required=False, default=[])
    parser.add_argument("-v", help="Output a vector file or not", action="store_true")
    args = parser.parse_args()
    main(args)

