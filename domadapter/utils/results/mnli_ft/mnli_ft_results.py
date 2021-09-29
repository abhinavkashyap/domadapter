import luigi
from domadapter.infer.glue_ft_infer import GlueFTInfer
import pathlib
import os
import csv
import numpy as np

RESULTS_DIR = pathlib.Path(os.environ["RESULTS_DIR"])
RESULTS_DIR = RESULTS_DIR.joinpath("mnlit_ft")


class MNLIResultForOneExperiment(luigi.Task):
    """
    Creates a CSV result file for one experiment of finetuning MNLI
    The CSV has the following columns
    Domain, Experiment Name, Accuracy
    Travel, exp_1, 0.79


    Parameters
    ==========
    experiments_folder: str
        The results of MNLI finetuning for a domain are assumed to be
        stored in a folder named by its domain such as `travel`. It further contains
        folders with an experiment as shown below
            travel/exp1
            travel/exp2
        where travel is a domain.
        The experiments folder is the /path/to/travel/exp_n
        The different experiments can correspond to different hyper-parameter
        trials

    """

    experiments_folder = luigi.Parameter()
    results_folder = luigi.Parameter()
    domain = luigi.Parameter()

    def run(self):
        experiments_folder = pathlib.Path(self.experiments_folder)
        infer = GlueFTInfer(experiments_dir=experiments_folder)
        results = infer.get_test_results()
        experiment_name = results.get("exp_name")
        acc = results.get("accuracy")
        with self.output().open("w") as fp:
            fieldnames = ["Domain", "Experiment", "Accuracy"]
            csvfile = csv.DictWriter(fp, fieldnames=fieldnames)
            csvfile.writeheader()
            csvfile.writerow(
                {"Domain": self.domain, "Experiment": experiment_name, "Accuracy": acc}
            )

    def output(self):
        # domain name = travel in /path/to/travel/exp1
        results_folder = pathlib.Path(self.results_folder)

        # extracts exp1 from the path
        exp_name = pathlib.Path(self.experiments_folder).name
        if not results_folder.is_dir():
            results_folder.mkdir(parents=True)

        return luigi.LocalTarget(path=results_folder.joinpath(f"{exp_name}.csv"))


class CollateMNLIMeanStdAccForOneDomain(luigi.Task):
    """
        Collects the results from all the experiments mentioned
        in a folder.  It outputs a file with the mean and standard
        deviation of the results of the experiments.
        You can use this to collate information about
        experiments with different seeds.
    """
    experiments_folder_domain = luigi.Parameter()
    results_folder = luigi.Parameter()
    output_filename = luigi.Parameter()
    domain = luigi.Parameter()

    def requires(self):
        domain_folder = pathlib.Path(self.experiments_folder_domain)

        return [
            MNLIResultForOneExperiment(experiments_folder=str(folder),
                                       results_folder=str(self.results_folder),
                                       domain=self.domain)
            for folder in domain_folder.iterdir()
        ]

    def run(self):
        accuracies = []
        for result_file in self.input():
            with result_file.open("r") as fp:
                csv_reader = csv.DictReader(fp)
                for row in csv_reader:
                    accuracies.append(float(row["Accuracy"]))

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        domain = pathlib.Path(self.experiments_folder_domain).name
        with self.output().open("w") as fp:
            fieldnames = ["Domain", "Acc(Mean)", "Acc(Std)"]
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {"Domain": domain, "Acc(Mean)": mean_accuracy, "Acc(Std)": std_accuracy}
            )

    def output(self):
        return luigi.LocalTarget(path=self.output_filename)


if __name__ == "__main__":
    domain = "travel"
    experiments_dir = "/ssd1/abhinav/domadapter/experiments/mnli_ft/travel/"
    results_folder = RESULTS_DIR.joinpath(domain)
    output_filename = results_folder.joinpath("agg_results.csv")

    luigi.build(
        [
            CollateMNLIMeanStdAccForOneDomain(
                experiments_folder_domain=experiments_dir,
                results_folder=str(results_folder),
                output_filename=str(output_filename),
                domain=domain
            )
        ],
        local_scheduler=True,
    )
