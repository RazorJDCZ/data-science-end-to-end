import pandas as pd

from monitoreo.desempeno import calcular_desempeno


def test_calcula_mae_por_semana_y_version():
    predicciones = pd.DataFrame({
        "id_prediccion": ["a", "b", "c"],
        "cod_canton": ["0101", "0101", "0101"],
        "fecha_objetivo": ["2026-09-06", "2026-09-07", "2026-09-08"],
        "version_modelo": ["1", "1", "2"],
        "emergencias_estimadas": [12.0, 15.0, 8.0],
    })
    verdad = pd.DataFrame({
        "cod_canton": ["0101", "0101", "0101"],
        "fecha": ["2026-09-06", "2026-09-07", "2026-09-08"],
        "n_emergencias": [10, 12, 10],
    })

    resultado = calcular_desempeno(predicciones, verdad)

    assert resultado.loc[(pd.Timestamp("2026-08-31").date(), "1"), "mae"] == 2.0
    assert resultado.loc[(pd.Timestamp("2026-09-07").date(), "1"), "mae"] == 3.0
    assert resultado.loc[(pd.Timestamp("2026-09-07").date(), "2"), "mae"] == 2.0
