import torch
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class TargetLabels:

    BAND_ENERGY = 'band_energy_ev'
    REPULSIVE_ENERGY = 'repulsive_energy_ev'
    TOTAL_ENERGY = 'total_energy_ev'
    MERMIN_FREE_ENERGY = 'mermin_free_energy_ev'
    FERMI_LEVEL = 'fermi_level_ev'

    E_HOMO = 'e_homo_ev'
    E_LUMO = 'e_lumo_ev'
    E_GAP = 'e_g_ev'
    MAX_OCCUPANCY = 'max_occupancy'
    METAL_LIKE = 'metal_like'

    IONIZATION_POTENTIAL = 'i_ev'
    ELECTRON_AFFINITY = 'a_ev'
    ELECTRONEGATIVITY = 'chi_ev'
    CHEMICAL_POTENTIAL = 'mu_ev'
    HARDNESS = 'eta_ev'
    SOFTNESS = 'softness_evinv'
    ELECTROPHILICITY = 'electrophilicity_ev'

    DIPOLE_X = 'dipole_x_d'
    DIPOLE_Y = 'dipole_y_d'
    DIPOLE_Z = 'dipole_z_d'
    DIPOLE_MAG = 'dipole_mag_d'

    TOTAL_CHARGE = 'total_charge'
    Q_ABSMEAN = 'q_absmean'
    Q_MAXABS = 'q_maxabs'
    Q_STD = 'q_std'

    DISTANCE = 'distance_ang'
    N_ATOMS = 'n_atoms'

    SCC_LAST_ITER = 'scc_last_iter'
    SCC_LAST_TOTAL_ELEC = 'scc_last_total_elec_eh'
    SCC_LAST_DIFF_ELEC = 'scc_last_diff_elec'
    SCC_LAST_ERROR = 'scc_last_error'
    SCC_CONVERGED = 'scc_converged'
    GEOM_CONVERGED = 'geom_converged'
    GEOM_OPT_STEP = 'geom_opt_step'

    N_LEVELS = 'n_levels'
    NO_VIRTUAL_IN_BASIS = 'no_virtual_in_basis'

    @classmethod
    def get_all(cls) -> List[str]:

        return [v for k, v in cls.__dict__.items()
                if not k.startswith('_') and isinstance(v, str) and not callable(v)]

    @classmethod
    def get_energy_targets(cls) -> List[str]:

        return [cls.BAND_ENERGY, cls.REPULSIVE_ENERGY, cls.TOTAL_ENERGY,
                cls.MERMIN_FREE_ENERGY, cls.FERMI_LEVEL]

    @classmethod
    def get_electronic_targets(cls) -> List[str]:

        return [cls.E_HOMO, cls.E_LUMO, cls.E_GAP, cls.MAX_OCCUPANCY, cls.METAL_LIKE]

    @classmethod
    def get_reactivity_targets(cls) -> List[str]:

        return [cls.IONIZATION_POTENTIAL, cls.ELECTRON_AFFINITY, cls.ELECTRONEGATIVITY,
                cls.CHEMICAL_POTENTIAL, cls.HARDNESS, cls.SOFTNESS, cls.ELECTROPHILICITY]

    @classmethod
    def get_dipole_targets(cls) -> List[str]:

        return [cls.DIPOLE_X, cls.DIPOLE_Y, cls.DIPOLE_Z, cls.DIPOLE_MAG]

    @classmethod
    def get_charge_targets(cls) -> List[str]:

        return [cls.TOTAL_CHARGE, cls.Q_ABSMEAN, cls.Q_MAXABS, cls.Q_STD]

    @classmethod
    def get_geometric_targets(cls) -> List[str]:

        return [cls.DISTANCE, cls.N_ATOMS]

    @classmethod
    def get_recommended_targets(cls) -> List[str]:

        return [cls.E_GAP, cls.TOTAL_ENERGY, cls.DIPOLE_MAG]

@dataclass
class ExperimentConfig:

    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])

    targets: List[str] = field(default_factory=lambda: [

        TargetLabels.E_GAP,
        TargetLabels.E_HOMO,
        TargetLabels.E_LUMO,
        TargetLabels.BAND_ENERGY,
        TargetLabels.REPULSIVE_ENERGY,
        TargetLabels.MERMIN_FREE_ENERGY,

        TargetLabels.TOTAL_ENERGY,
        TargetLabels.IONIZATION_POTENTIAL,
        TargetLabels.ELECTRON_AFFINITY,

        TargetLabels.ELECTRONEGATIVITY,
        TargetLabels.CHEMICAL_POTENTIAL,
        TargetLabels.HARDNESS,
        TargetLabels.SOFTNESS,
        TargetLabels.ELECTROPHILICITY,

        TargetLabels.DIPOLE_MAG,
        TargetLabels.DIPOLE_Z,

        TargetLabels.DISTANCE,

        TargetLabels.Q_MAXABS,
        TargetLabels.Q_ABSMEAN,
        TargetLabels.Q_STD,
    ])

    @property
    def total_experiments(self) -> int:
                return len(self.seeds) * len(self.targets)

@dataclass
class DataConfig:

    dataset_path: str = 'dataset_combined.npz'
    train_split: float = 0.8
    val_split: float = 0.1
    num_workers: int = 4

    @property
    def test_split(self) -> float:

        return 1.0 - self.train_split - self.val_split

@dataclass
class TrainingConfig:

    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    early_stopping: bool = True
    shuffle: bool = True

@dataclass
class OptimizerConfig:

    name: str = 'adam'
    params: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'adam': {},
        'adamw': {'betas': (0.9, 0.999)},
        'sgd': {'momentum': 0.9}
    })

    def get_params(self) -> Dict[str, Any]:

        return self.params.get(self.name, {})

@dataclass
class SchedulerConfig:

    name: str = 'reduce_on_plateau'
    params: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'reduce_on_plateau': {
            'mode': 'min',
            'factor': 0.8,
            'patience': 10,
            'min_lr': 1e-6
        },
        'cosine': {
            'T_max': 5,
            'eta_min': 1e-6
        },
        'step': {
            'step_size': 50,
            'gamma': 0.5
        }
    })

    def get_params(self) -> Dict[str, Any]:

        return self.params.get(self.name, {})

    def update_epochs(self, epochs: int):

        if 'cosine' in self.params:
            self.params['cosine']['T_max'] = epochs

@dataclass
class LossConfig:

    name: str = 'mae'

@dataclass
class ModelConfigs:

    def __init__(self):
        self.schnet = {
            'hidden_channels': 96,
            'num_filters': 96,
            'num_interactions': 6,
            'num_gaussians': 50,
            'cutoff': 5.0
        }

        self.faenet = {
            'cutoff': 5.0,
            'hidden_channels': 128,
            'num_filters': 128,
            'num_interactions': 4
        }

        self.gotennet = {
            'n_atom_basis': 64,
            'n_interactions': 3,
            'cutoff': 5.0,
            'num_heads': 2,
            'n_rbf': 10
        }

        self.egnn = {
            'n_layers': 7,
            'feats_dim': 1,
            'pos_dim': 3,
            'm_dim': 180,
            'update_coors': True,
            'update_feats': True,
            'norm_feats': True,
            'norm_coors': False,
            'dropout': 0.0,
            'coor_weights_clamp_value': 2.0
        }

        self.gatv2 = {
            'hidden_channels': 96,
            'num_layers': 5,
            'heads': 4,
            'dropout': 0.1
        }

        self.dimenet = {
            'hidden_channels': 56,
            'out_channels': 1,
            'num_blocks': 3,
            'int_emb_size': 56,
            'basis_emb_size': 8,
            'out_emb_channels': 112,
            'num_spherical': 7,
            'num_radial': 6,
            'cutoff': 5.0,
            'envelope_exponent': 5,
            'num_before_skip': 1,
            'num_after_skip': 2,
            'num_output_layers': 3
        }

        self.vit = {
            'patch_size': 4,
            'embed_dim': 96,
            'num_heads': 4,
            'num_layers': 4
        }

        self.quantumshellnet = {}

        self.multimodal = {
            'hidden_channels': 96,
            'num_filters': 96,
            'num_interactions': 6,
            'num_gaussians': 50,
            'cutoff': 5.0,
            'image_features_dim': 512,
            'mlp_hidden_dims': [256, 128, 64]
        }

    def get(self, model_name: str) -> Dict[str, Any]:

        return getattr(self, model_name, {})

    def to_dict(self) -> Dict[str, Dict[str, Any]]:

        return {
            'schnet': self.schnet,
            'faenet': self.faenet,
            'gotennet': self.gotennet,
            'egnn': self.egnn,
            'gatv2': self.gatv2,
            'dimenet': self.dimenet,
            'vit': self.vit,
            'quantumshellnet': self.quantumshellnet,
            'multimodal': self.multimodal
        }

    @property
    def available_models(self) -> List[str]:

        return ['schnet', 'faenet', 'gotennet', 'egnn', 'gatv2', 'dimenet',
                'vit', 'quantumshellnet', 'multimodal']

@dataclass
class BenchmarkConfig:

    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    models: ModelConfigs = field(default_factory=ModelConfigs)

    def __post_init__(self):

        self.scheduler.update_epochs(self.training.epochs)

    @property
    def seeds(self) -> List[int]:

        return self.experiment.seeds

    @property
    def targets(self) -> List[str]:

        return self.experiment.targets

    @property
    def batch_size(self) -> int:

        return self.training.batch_size

    @property
    def epochs(self) -> int:

        return self.training.epochs

    @property
    def lr(self) -> float:

        return self.training.lr

    @property
    def dataset_path(self) -> str:

        return self.data.dataset_path

    @property
    def train_split(self) -> float:

        return self.data.train_split

    @property
    def val_split(self) -> float:

        return self.data.val_split

    @property
    def test_split(self) -> float:

        return self.data.test_split

    @property
    def num_workers(self) -> int:

        return self.data.num_workers

    @property
    def loss_function(self) -> str:

        return self.loss.name

    @property
    def total_experiments(self) -> int:

        return self.experiment.total_experiments

    @property
    def available_models(self) -> List[str]:

        return self.models.available_models

    @property
    def model_configs(self) -> Dict[str, Dict[str, Any]]:

        return self.models.to_dict()

    @property
    def optimizer_name(self) -> str:

        return self.optimizer.name

    @property
    def optimizer_params(self) -> Dict[str, Dict[str, Any]]:

        return self.optimizer.params

    @property
    def scheduler_name(self) -> str:

        return self.scheduler.name

    @property
    def scheduler_params(self) -> Dict[str, Dict[str, Any]]:

        return self.scheduler.params

    @property
    def training_params(self) -> Dict[str, Any]:

        return {
            'patience': self.training.patience,
            'early_stopping': self.training.early_stopping,
            'shuffle': self.training.shuffle,
            'weight_decay': self.training.weight_decay
        }

    def get_model_config(self, model_name: str) -> Dict[str, Any]:

        return self.models.get(model_name)

    def to_dict(self) -> Dict[str, Any]:

        return {
            'device': self.device,
            'seeds': self.experiment.seeds,
            'targets': self.experiment.targets,
            'dataset_path': self.data.dataset_path,
            'train_split': self.data.train_split,
            'val_split': self.data.val_split,
            'test_split': self.data.test_split,
            'num_workers': self.data.num_workers,
            'batch_size': self.training.batch_size,
            'epochs': self.training.epochs,
            'lr': self.training.lr,
            'weight_decay': self.training.weight_decay,
            'patience': self.training.patience,
            'early_stopping': self.training.early_stopping,
            'optimizer': self.optimizer_name,
            'scheduler': self.scheduler_name,
            'loss': self.loss_function,
        }

    def print_summary(self):

        print("=" * 70)
        print("BENCHMARK CONFIGURATION SUMMARY")
        print("=" * 70)

        print("\n🎯 Experiment Design:")
        print(f"  Targets: {len(self.experiment.targets)} ({', '.join(self.experiment.targets)})")
        print(f"  Seeds: {len(self.experiment.seeds)} ({', '.join(map(str, self.experiment.seeds))})")
        print(f"  Total experiments: {self.experiment.total_experiments}")
        print(f"  Available models: {len(self.models.available_models)}")

        print("\n⚙️  Training Settings:")
        print(f"  Batch size: {self.training.batch_size}")
        print(f"  Epochs: {self.training.epochs}")
        print(f"  Learning rate: {self.training.lr}")
        print(f"  Weight decay: {self.training.weight_decay}")
        print(f"  Early stopping: {self.training.early_stopping} (patience={self.training.patience})")
        print(f"  Optimizer: {self.optimizer_name}")
        print(f"  Scheduler: {self.scheduler_name}")
        print(f"  Loss: {self.loss_function}")
        print(f"  Device: {self.device}")

        print("\n📊 Data Configuration:")
        print(f"  Dataset: {self.data.dataset_path}")
        print(f"  Train/Val/Test split: {self.data.train_split}/{self.data.val_split}/{self.data.test_split:.2f}")
        print(f"  Num workers: {self.data.num_workers}")

        print("\n🤖 Models:")
        for model_name in self.models.available_models:
            print(f"  - {model_name}")
        print("=" * 70)

cfg = BenchmarkConfig()
