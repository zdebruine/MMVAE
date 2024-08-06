import csv
import subprocess

def get_explicit_packages():
    # Get the list of explicitly installed packages using pip list --not-required
    result = subprocess.run(['pip', 'list', '--not-required', '--format=freeze'], capture_output=True, text=True)
    packages = result.stdout.strip().split('\n')
    return packages

def write_csv(packages, filename='requirements.csv'):
    # Write the packages to a CSV file
    with open(filename, mode='w', newline='') as csvfile:
        fieldnames = ['package', 'version']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for package in packages:
            name, version = package.split('==')
            writer.writerow({'package': name, 'version': version})

def main():
    packages = get_explicit_packages()
    write_csv(packages)
    print("requirements.csv has been generated with the explicitly installed packages.")

if __name__ == '__main__':
    main()