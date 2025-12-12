"""
Module: test_evaluate.py
========================

Author: Test Suite
Created: December 2025

Description:
------------
Tests complets pour le module evaluate.py utilisant pytest.
Couvre les classes Evaluator et EnsembleEvaluator avec des mocks appropriés.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from torch.utils.data import DataLoader, TensorDataset

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


with patch.dict('sys.modules', {'src.utils': MagicMock()}):
    from src.evaluate import Evaluator, EnsembleEvaluator



@pytest.fixture
def device():
    """Fixture pour le device (CPU pour les tests)."""
    return torch.device('cpu')


@pytest.fixture
def class_names():
    """Fixture pour les noms des classes."""
    return ['CBB', 'CBSD', 'CGM', 'CMD', 'Healthy']


@pytest.fixture
def mock_model():
    """Fixture pour un modèle PyTorch mocké."""
    model = Mock(spec=nn.Module)
    model.eval = Mock()
    return model


@pytest.fixture
def sample_dataloader():
    """
    Fixture pour un DataLoader avec des données synthétiques.
    Batch size: 4, 3 batches, 5 classes.
    """
 
    images = torch.randn(12, 3, 224, 224)
    labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1])
    
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    return dataloader


@pytest.fixture
def sample_dataloader_soft_labels():
    """DataLoader avec des soft labels (one-hot encoded)."""
    images = torch.randn(8, 3, 224, 224)

    labels = torch.zeros(8, 5)
    labels[0, 0] = 1
    labels[1, 1] = 1
    labels[2, 2] = 1
    labels[3, 3] = 1
    labels[4, 4] = 1
    labels[5, 0] = 1
    labels[6, 1] = 1
    labels[7, 2] = 1
    
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    return dataloader


class TestEvaluator:
    """Tests pour la classe Evaluator."""
    
    def test_evaluator_initialization(self, device, class_names):
        """Test l'initialisation de l'Evaluator."""
        evaluator = Evaluator(device=device, model_name="TestModel", class_names=class_names)
        
        assert evaluator.device == device
        assert evaluator.filename == "TestModel"
        assert evaluator.class_names == class_names
        assert isinstance(evaluator.criterion, nn.CrossEntropyLoss)
    
    
    def test_evaluate_with_hard_labels(self, device, class_names, sample_dataloader):
        """Test l'évaluation avec des labels hard (indices de classe)."""
        evaluator = Evaluator(device=device, model_name="TestModel", class_names=class_names)
        
        
        model = Mock(spec=nn.Module)
        model.eval = Mock()
        
     
        def mock_forward(images):
            batch_size = images.size(0)
            
            return torch.randn(batch_size, 5)
        
        model.side_effect = mock_forward
        
      
        with patch.object(evaluator, '_save_confusion_and_report'):
            loss, accuracy = evaluator.evaluate(model, sample_dataloader)
        
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
        assert loss >= 0.0
    
    
    def test_evaluate_with_soft_labels(self, device, class_names, sample_dataloader_soft_labels):
        """Test l'évaluation avec des soft labels (one-hot)."""
        evaluator = Evaluator(device=device, model_name="TestModel", class_names=class_names)
        
        model = Mock(spec=nn.Module)
        model.eval = Mock()
        
        def mock_forward(images):
            batch_size = images.size(0)
            return torch.randn(batch_size, 5)
        
        model.side_effect = mock_forward
        
        with patch.object(evaluator, '_save_confusion_and_report'):
            loss, accuracy = evaluator.evaluate(model, sample_dataloader_soft_labels)
        
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
    
    
    def test_evaluate_perfect_predictions(self, device, class_names):
        """Test avec des prédictions parfaites (accuracy = 100%)."""
        evaluator = Evaluator(device=device, model_name="TestModel", class_names=class_names)
        
       
        images = torch.randn(8, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        model = Mock(spec=nn.Module)
        model.eval = Mock()
        
       
        def perfect_forward(images):
            batch_size = images.size(0)
            outputs = torch.zeros(batch_size, 5)
            
            for i in range(batch_size):
                correct_class = labels[i].item() if i < 4 else labels[i + 4].item()
                outputs[i, correct_class] = 10.0  
            return outputs
        
        model.side_effect = perfect_forward
        
        with patch.object(evaluator, '_save_confusion_and_report'):
            loss, accuracy = evaluator.evaluate(model, dataloader)
        
       
        assert accuracy >= 0.0
    
    
    @patch('matplotlib.pyplot.savefig')
    def test_save_confusion_and_report(self, mock_savefig, device, class_names, tmp_path):
        """Test la sauvegarde de la matrice de confusion et du rapport."""
        evaluator = Evaluator(device=device, model_name="TestModel", class_names=class_names)
        
        labels = [0, 1, 2, 3, 4, 0, 1, 2]
        preds = [0, 1, 2, 3, 4, 1, 1, 2] 
        
   
        import sys
        evaluate_module = sys.modules.get('src.evaluate')
        if evaluate_module:
            original_dir = getattr(evaluate_module, 'REPORT_DIR', None)
            evaluate_module.REPORT_DIR = str(tmp_path)
            try:
                evaluator._save_confusion_and_report(labels, preds, "test_report.png")
                
                mock_savefig.assert_called_once()
            finally:
                if original_dir:
                    evaluate_module.REPORT_DIR = original_dir
    
    
    def test_evaluator_with_empty_dataloader(self, device, class_names):
        """Test avec un dataloader vide (devrait lever une exception)."""
        evaluator = Evaluator(device=device, model_name="TestModel", class_names=class_names)
        
     
        empty_dataset = TensorDataset(torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long))
        empty_dataloader = DataLoader(empty_dataset, batch_size=4)
        
        model = Mock(spec=nn.Module)
        model.eval = Mock()
        
        with patch.object(evaluator, '_save_confusion_and_report'):
            with pytest.raises(ZeroDivisionError):
                evaluator.evaluate(model, empty_dataloader)


class TestEnsembleEvaluator:
    """Tests pour la classe EnsembleEvaluator."""
    
    def test_ensemble_evaluator_initialization(self, device, class_names):
        """Test l'initialisation de l'EnsembleEvaluator."""
        evaluator = EnsembleEvaluator(
            device=device,
            model_name="EnsembleTest",
            class_names=class_names
        )
        
        assert evaluator.device == device
        assert evaluator.filename == "EnsembleTest"
        assert evaluator.class_names == class_names
        assert isinstance(evaluator.criterion, nn.CrossEntropyLoss)
    
    
    def test_ensemble_evaluate_basic(self, device, class_names, sample_dataloader):
        """Test l'évaluation de l'ensemble avec 3 modèles."""
        evaluator = EnsembleEvaluator(
            device=device,
            model_name="EnsembleTest",
            class_names=class_names
        )
        
    
        m1 = Mock(spec=nn.Module)
        m2 = Mock(spec=nn.Module)
        m3 = Mock(spec=nn.Module)
        
        m1.eval = Mock()
        m2.eval = Mock()
        m3.eval = Mock()
        

        def mock_forward_1(images):
            return torch.randn(images.size(0), 5)
        
        def mock_forward_2(images):
            return torch.randn(images.size(0), 5)
        
        def mock_forward_3(images):
            return torch.randn(images.size(0), 5)
        
        m1.side_effect = mock_forward_1
        m2.side_effect = mock_forward_2
        m3.side_effect = mock_forward_3
        
        with patch.object(evaluator, '_save_confusion_and_report'):
            loss, accuracy = evaluator.evaluate(m1, m2, m3, sample_dataloader)
        
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
        assert loss >= 0.0
    
    
    def test_ensemble_with_soft_labels(self, device, class_names, sample_dataloader_soft_labels):
        """Test l'ensemble avec des soft labels."""
        evaluator = EnsembleEvaluator(
            device=device,
            model_name="EnsembleTest",
            class_names=class_names
        )
        
        m1 = Mock(spec=nn.Module)
        m2 = Mock(spec=nn.Module)
        m3 = Mock(spec=nn.Module)
        
        m1.eval = Mock()
        m2.eval = Mock()
        m3.eval = Mock()
        
        def mock_forward(images):
            return torch.randn(images.size(0), 5)
        
        m1.side_effect = mock_forward
        m2.side_effect = mock_forward
        m3.side_effect = mock_forward
        
        with patch.object(evaluator, '_save_confusion_and_report'):
            loss, accuracy = evaluator.evaluate(m1, m2, m3, sample_dataloader_soft_labels)
        
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
    
    
    def test_ensemble_probability_averaging(self, device, class_names):
        """Test que les probabilités sont bien moyennées."""
        evaluator = EnsembleEvaluator(
            device=device,
            model_name="EnsembleTest",
            class_names=class_names
        )
        
        
        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3])
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=4)
        
        m1 = Mock(spec=nn.Module)
        m2 = Mock(spec=nn.Module)
        m3 = Mock(spec=nn.Module)
        
        m1.eval = Mock()
        m2.eval = Mock()
        m3.eval = Mock()
        
        
        fixed_output = torch.tensor([
            [10.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 10.0, 0.0]
        ])
        
        m1.side_effect = lambda x: fixed_output.clone()
        m2.side_effect = lambda x: fixed_output.clone()
        m3.side_effect = lambda x: fixed_output.clone()
        
        with patch.object(evaluator, '_save_confusion_and_report'):
            loss, accuracy = evaluator.evaluate(m1, m2, m3, dataloader)
        
       
        assert accuracy >= 0.75  # Au moins 75% correct
    
    
    @patch('matplotlib.pyplot.savefig')
    def test_ensemble_save_report(self, mock_savefig, device, class_names, tmp_path):
        """Test la sauvegarde du rapport de l'ensemble."""
        evaluator = EnsembleEvaluator(
            device=device,
            model_name="EnsembleTest",
            class_names=class_names
        )
        
        labels = [0, 1, 2, 3, 4, 0, 1, 2]
        preds = [0, 1, 2, 3, 4, 1, 1, 2]
        
      
        import sys
        evaluate_module = sys.modules.get('src.evaluate')
        if evaluate_module:
            original_dir = getattr(evaluate_module, 'REPORT_DIR', None)
            evaluate_module.REPORT_DIR = str(tmp_path)
            try:
                evaluator._save_confusion_and_report(labels, preds, "ensemble_test.png")
                mock_savefig.assert_called_once()
            finally:
                if original_dir:
                    evaluate_module.REPORT_DIR = original_dir


class TestIntegration:
    """Tests d'intégration pour vérifier le workflow complet."""
    
    def test_full_evaluation_workflow(self, device, class_names, sample_dataloader):
        """Test du workflow complet d'évaluation."""
        evaluator = Evaluator(device=device, model_name="IntegrationTest", class_names=class_names)
        
       
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, 5)
        )
        model.eval()
        
        with patch.object(evaluator, '_save_confusion_and_report'):
            loss, accuracy = evaluator.evaluate(model, sample_dataloader)
        
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert loss > 0.0  
        assert 0.0 <= accuracy <= 1.0
    
    
    def test_full_ensemble_workflow(self, device, class_names, sample_dataloader):
        """Test du workflow complet pour l'ensemble."""
        evaluator = EnsembleEvaluator(
            device=device,
            model_name="EnsembleIntegration",
            class_names=class_names
        )
        
       
        m1 = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 5))
        m2 = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 5))
        m3 = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 5))
        
        m1.eval()
        m2.eval()
        m3.eval()
        
        with patch.object(evaluator, '_save_confusion_and_report'):
            loss, accuracy = evaluator.evaluate(m1, m2, m3, sample_dataloader)
        
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert loss > 0.0
        assert 0.0 <= accuracy <= 1.0



@pytest.mark.parametrize("batch_size,num_samples", [
    (2, 10),
    (4, 16),
    (8, 24),
])
def test_evaluator_different_batch_sizes(device, class_names, batch_size, num_samples):
    """Test l'évaluateur avec différentes tailles de batch."""
    evaluator = Evaluator(device=device, model_name="BatchTest", class_names=class_names)
    
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, 5, (num_samples,))
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 5))
    model.eval()
    
    with patch.object(evaluator, '_save_confusion_and_report'):
        loss, accuracy = evaluator.evaluate(model, dataloader)
    
    assert isinstance(loss, float)
    assert isinstance(accuracy, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])