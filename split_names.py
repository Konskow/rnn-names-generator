def save_list(path, list):
    with open(path, 'w') as f:
        for name in sorted(list):
            f.write(name)


if __name__ == "__main__":
    female_names = []
    male_names = []

    with open('data/polish/forenames.txt', 'r') as file:

        for line in file:
            clean_line = line.strip('\n\r ')
            if len(clean_line) > 0:
                if clean_line[-1] == 'a':
                    female_names.append(line)
                else:
                    male_names.append(line)

    save_list('data/polish/female.txt', female_names)
    save_list('data/polish/male.txt', male_names)
